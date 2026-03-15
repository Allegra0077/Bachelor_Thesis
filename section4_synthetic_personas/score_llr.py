"""
Role-based log-likelihood ratio scoring for synthetic persona conversations (Section 4.3).

Scores each target user message under two role prompts using the same model
checkpoint: a user-role prompt and an assistant-role prompt. The log-likelihood
ratio (LLR) measures how much more compatible the message is with the user role
than the assistant role. Positive LLR indicates user-role compatibility.

For the hidden-persona condition, the user-role scorer receives the same persona
information that was used during generation, matching the generation context.

The context provided to both scorers consists of the immediately preceding
assistant message, matching the context available to the user model during
generation.

Example usage:
    python score_llr.py \
        --input_path conversations.jsonl \
        --assistant_model Qwen/Qwen3-4B-Instruct-2507 \
        --output_path llr_scores.jsonl

Output format (JSONL, one row per target user message):
    {
        "conversation_id": "uuid",
        "condition": "no_persona" | "hidden_persona",
        "target_user_msg_idx": 4,
        "avg_logprob_assistant_role": -3.12,
        "avg_logprob_user_role": -2.45,
        "llr_avg_logprob": 0.67,
        ...
    }
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- Default role prompts (same as used in generation) ----------
ASSISTANT_SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant."

USER_SYSTEM_PROMPT_DEFAULT = (
    "Respond as a user reacting naturally to the assistant's last message.\n\n"
    "Your response should be short (1–2 sentences) and conversational.\n"
    "You may:\n"
    "- ask for clarification\n"
    "- request a change or refinement\n"
    "- express agreement or disagreement\n"
    "- ask a follow-up question\n\n"
    "Do NOT:\n"
    "- continue the assistant's answer\n"
    "- add new content or solutions\n"
    "- restate the original task\n\n"
    "Write only the user's next message."
)


def build_user_context_prompt(
    condition: str, persona_text: Optional[str], base_user_prompt: str,
) -> str:
    """Build the user-role system prompt, including persona if applicable."""
    if condition == "hidden_persona" and persona_text:
        return (
            f"{base_user_prompt}\n\n"
            "=== INTERNAL CHARACTER NOTES (DO NOT MENTION THESE) ===\n"
            f"{persona_text}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- These are background facts about you as a person\n"
            "- They should SUBTLY influence your tone, interests, and reactions\n"
            "- DO NOT explicitly reference these facts unless the assistant directly asks about them\n"
            "- NEVER say things like 'my girlfriend' or 'I make 50k' unprompted\n"
            "- Stay focused on reacting to what the ASSISTANT just said\n"
        )
    return base_user_prompt


def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True, help="JSONL conversations file")
    p.add_argument("--output_path", type=str, default=f"llr_scores_{timestamp}.jsonl", help="JSONL output scores file")
    p.add_argument("--assistant_model", type=str, required=True, help="Assistant scoring checkpoint")
    p.add_argument("--user_model", type=str, default=None, help="User scoring checkpoint (default: same as assistant_model)")
    p.add_argument("--assistant_system_prompt", type=str, default=ASSISTANT_SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--user_system_prompt", type=str, default=USER_SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--prepend_system", default=True, action=argparse.BooleanOptionalAction, help="Prepend a system message before scoring")
    p.add_argument("--max_conversations", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


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
    """Return indices of user messages after the seed prompt."""
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    return user_idxs[1:]


def build_generation_match_context(
    messages: List[Dict[str, str]], target_user_idx: int,
) -> List[Dict[str, str]]:
    """
    Build context matching what the user model saw during generation:
    only the immediately preceding assistant message.
    """
    if target_user_idx - 1 >= 0 and messages[target_user_idx - 1].get("role") == "assistant":
        return [{"role": "assistant", "content": messages[target_user_idx - 1].get("content", "")}]
    return []


@torch.no_grad()
def score_target_user_text(
    model,
    tokenizer,
    context_messages: List[Dict[str, str]],
    target_user_text: str,
    system_prompt: str,
    prepend_system: bool,
) -> Tuple[float, float, int]:
    """
    Compute average and total log-probability for the target user message
    under the given system prompt.
    """
    ctx = context_messages
    if prepend_system:
        ctx = [{"role": "system", "content": system_prompt}] + ctx

    full = ctx + [{"role": "user", "content": target_user_text}]

    ctx_ids = apply_chat_template(tokenizer, ctx).to(model.device)
    full_ids = apply_chat_template(tokenizer, full).to(model.device)

    Lc = ctx_ids.shape[1]
    L = full_ids.shape[1]
    if L <= Lc:
        return float("nan"), float("nan"), 0

    out = model(full_ids)
    logits = out.logits

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

    if token_lp.numel() == 0:
        return float("nan"), float("nan"), 0

    avg_lp = token_lp.mean().item()
    total_lp = token_lp.sum().item()
    return avg_lp, total_lp, int(token_lp.numel())


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype):
    """Load model and tokenizer for scoring."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tok


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    dtype = _dtype_from_str(args.dtype)

    user_model_name = args.user_model or args.assistant_model
    same_checkpoint = (user_model_name == args.assistant_model)

    assistant_model, assistant_tok = load_model_and_tokenizer(args.assistant_model, dtype)

    if same_checkpoint:
        user_model, user_tok = assistant_model, assistant_tok
    else:
        user_model, user_tok = load_model_and_tokenizer(user_model_name, dtype)

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
            condition = conv.get("condition", "")
            persona_text = conv.get("persona_text")

            user_context_prompt = build_user_context_prompt(
                condition=condition,
                persona_text=persona_text,
                base_user_prompt=args.user_system_prompt,
            )

            user_idxs = get_user_message_indices(messages)

            for ui in user_idxs:
                target_text = messages[ui].get("content", "")
                ctx = build_generation_match_context(messages, target_user_idx=ui)

                # Score under assistant-role prompt
                a_avg_lp, a_total_lp, a_n_tok = score_target_user_text(
                    model=assistant_model,
                    tokenizer=assistant_tok,
                    context_messages=ctx,
                    target_user_text=target_text,
                    system_prompt=args.assistant_system_prompt,
                    prepend_system=args.prepend_system,
                )

                # Score under user-role prompt (includes persona if hidden_persona)
                u_avg_lp, u_total_lp, u_n_tok = score_target_user_text(
                    model=user_model,
                    tokenizer=user_tok,
                    context_messages=ctx,
                    target_user_text=target_text,
                    system_prompt=user_context_prompt,
                    prepend_system=args.prepend_system,
                )

                # Compute log-likelihood ratio
                llr_avg = (
                    u_avg_lp - a_avg_lp
                    if (u_avg_lp == u_avg_lp and a_avg_lp == a_avg_lp)
                    else float("nan")
                )
                llr_total = (
                    u_total_lp - a_total_lp
                    if (u_total_lp == u_total_lp and a_total_lp == a_total_lp)
                    else float("nan")
                )

                row = {
                    "conversation_id": conv.get("conversation_id"),
                    "condition": condition,
                    "persona_text": persona_text,
                    "seed_prompt": conv.get("seed_prompt"),
                    "target_user_msg_idx": ui,
                    "context_mode": "generation_match",
                    "assistant_model": args.assistant_model,
                    "user_model": user_model_name,
                    "same_checkpoint": same_checkpoint,
                    "avg_logprob_assistant_role": a_avg_lp,
                    "total_logprob_assistant_role": a_total_lp,
                    "n_tokens_assistant_role": a_n_tok,
                    "avg_logprob_user_role": u_avg_lp,
                    "total_logprob_user_role": u_total_lp,
                    "n_tokens_user_role": u_n_tok,
                    "llr_avg_logprob": llr_avg,
                    "llr_total_logprob": llr_total,
                }

                if a_n_tok != u_n_tok:
                    row["warning"] = "token_count_mismatch_between_roles"

                fout.write(json.dumps(row) + "\n")
                rows_written += 1

    print(f"Done. conversations={conv_count} rows={rows_written} -> {args.output_path}")


if __name__ == "__main__":
    main()