"""
NNLL scoring for synthetic persona conversations (Section 4.3).

Scores each target user message (excluding the seed prompt) under varying
amounts of preceding dialogue context. For each conversation, the script
iterates over non-seed user turns and computes the average log-probability
of the target message tokens given the context.

Context depth is controlled by --n_turns_list: the number of preceding
turns retained before the target message (0 = system prompt only).

Example usage:
    python score_nnll.py \
        --input_path conversations.jsonl \
        --assistant_model Qwen/Qwen3-4B-Instruct-2507 \
        --n_turns_list 0 1 2 3 4 6 \
        --output_path nnll_scores.jsonl

Output format (JSONL, one row per target-message x context-depth combination):
    {
        "conversation_id": "uuid",
        "condition": "no_persona" | "hidden_persona",
        "persona_text": "persona string or null",
        "seed_prompt": "initial user message",
        "target_user_msg_idx": 4,
        "n_turns_context": 2,
        "avg_logprob": -2.31,
        "n_tokens": 15
    }
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True, help="JSONL conversations file")
    parser.add_argument("--output_path", type=str, default=f"nnll_scores_{timestamp}.jsonl", help="JSONL output scores file")
    parser.add_argument("--assistant_model", type=str, required=True, help="Model used for scoring")
    parser.add_argument("--n_turns_list", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6])
    parser.add_argument("--prepend_system", type=bool, default=True, help="Prepend a system message before scoring")
    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _dtype_from_str(s: str):
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> torch.Tensor:
    """Returns input_ids shaped (1, L). No generation prompt since we are scoring."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )


def get_user_message_indices(messages: List[Dict[str, str]]) -> List[int]:
    """
    Return indices of user messages after the seed prompt.
    The first user message is the seed and is skipped.
    """
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    return user_idxs[1:]


def truncate_context_keep_last_n_turns(
    messages: List[Dict[str, str]],
    target_user_idx: int,
    n_turns: int,
) -> List[Dict[str, str]]:
    """
    Keep last n turns of context before the target user message.
    A turn boundary is defined at each assistant message.

    n_turns=0 => no transcript context (system prompt only)
    n_turns>=1 => walk backwards until n assistant messages are included
    """
    if n_turns == 0:
        return []

    prefix = messages[:target_user_idx]
    kept_rev: List[Dict[str, str]] = []

    turns = 0
    for m in reversed(prefix):
        kept_rev.append(m)
        if m["role"] == "assistant":
            turns += 1
            if turns >= n_turns:
                break

    return list(reversed(kept_rev))


@torch.no_grad()
def score_target_user_text(
    model,
    tokenizer,
    context_messages: List[Dict[str, str]],
    target_user_text: str,
    prepend_system: bool,
) -> Tuple[float, int]:
    """
    Compute average log-probability per token for the target user message,
    conditioned on the provided context.
    """
    ctx = context_messages
    if prepend_system:
        ctx = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + ctx

    full = ctx + [{"role": "user", "content": target_user_text}]

    ctx_ids = apply_chat_template(tokenizer, ctx).to(model.device)
    full_ids = apply_chat_template(tokenizer, full).to(model.device)

    Lc = ctx_ids.shape[1]
    L = full_ids.shape[1]
    if L <= Lc:
        return float("nan"), 0

    out = model(full_ids)
    logits = out.logits

    # Target token positions: [Lc, ..., L-1]
    # Corresponding logits positions: [Lc-1, ..., L-2]
    start = max(Lc - 1, 0)
    end = L - 1

    logits_slice = logits[0, start:end, :]
    target_ids = full_ids[0, Lc:L]

    if logits_slice.shape[0] != target_ids.shape[0]:
        T = min(logits_slice.shape[0], target_ids.shape[0])
        logits_slice = logits_slice[:T]
        target_ids = target_ids[:T]

    log_probs = F.log_softmax(logits_slice, dim=-1)
    token_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    avg_lp = token_lp.mean().item() if token_lp.numel() > 0 else float("nan")
    return avg_lp, int(token_lp.numel())


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.assistant_model)

    dtype = _dtype_from_str(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    rows_written = 0
    conv_count = 0

    with open(args.input_path, "r") as fin, open(args.output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            conv = json.loads(line)
            conv_count += 1
            if args.max_conversations is not None and conv_count > args.max_conversations:
                break

            messages = conv["messages"]
            user_idxs = get_user_message_indices(messages)

            for ui in user_idxs:
                target_text = messages[ui]["content"]

                for n_turns in args.n_turns_list:
                    ctx = truncate_context_keep_last_n_turns(
                        messages, target_user_idx=ui, n_turns=n_turns,
                    )

                    avg_lp, n_tok = score_target_user_text(
                        model=model,
                        tokenizer=tok,
                        context_messages=ctx,
                        target_user_text=target_text,
                        prepend_system=args.prepend_system,
                    )

                    row = {
                        "conversation_id": conv.get("conversation_id"),
                        "condition": conv.get("condition"),
                        "persona_text": conv.get("persona_text"),
                        "seed_prompt": conv.get("seed_prompt"),
                        "target_user_msg_idx": ui,
                        "n_turns_context": n_turns,
                        "avg_logprob": avg_lp,
                        "n_tokens": n_tok,
                    }
                    fout.write(json.dumps(row) + "\n")
                    rows_written += 1

    print(f"Done. conversations={conv_count} rows={rows_written} -> {args.output_path}")


if __name__ == "__main__":
    main()