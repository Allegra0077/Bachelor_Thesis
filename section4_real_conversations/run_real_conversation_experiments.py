"""
Scoring script for real multi-turn conversation experiments (Section 4.2).

This script scores the final user message or assistant response under
varying dialogue contexts using LMSYS-Chat-1M conversations.

Experiment flags (set at bottom of file):
- HUMAN_ONLY / MODEL_ONLY: context type (HO, MO, or full)
- TEST_HUMAN_MESSAGE / TEST_MODEL_MESSAGE: which message to score
- INJECT_RANDOM_TOPIC: negative control (Section 4.2.3)
- MIX_INPUTS: permuted user history (Section 4.2.3)
- INVESTIGATOR_SETTING: investigator system prompt (Section 4.2.3)
- LOW_TOKENS: filter out conversations exceeding 4000 tokens
"""

import os
import json
import time
import random

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

random.seed(42)

INVESTIGATOR_PROMPT = (
    "You are an expert investigator. As you hold a helpful conversation with the user, "
    "you should focus on extracting as much detailed information out of them as possible. "
    "Your final goal is to predict the user's question as they are asking it."
)

# seq is Qwen3 Tok of "<think>\n\n</think>\n\n"
def remove_sequence_list(data, seq=[151667, 271, 151668, 271]):
    n = len(seq)
    for i in range(len(data) - n + 1):
        if data[i : i + n] == seq:
            return data[:i] + data[i + n:]
    return data


def apply_template(conversation, tokenizer, remove_think_tokens=False):
    """Apply the default chat template to a conversation."""
    assert conversation[0]["role"] == "system"

    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        enable_thinking=False,
    )

    inputs["input_ids"] = (
        torch.tensor([remove_sequence_list(inputs["input_ids"])])
        if remove_think_tokens
        else torch.tensor([inputs["input_ids"]])
    )
    return inputs


def main(max_turns):

    print("Starting multi-turn experiment")
    print("=" * 20)
    start = time.time()

    human_first_turn = not MODEL_ONLY

    assert not (HUMAN_ONLY and MODEL_ONLY), "Cannot set both HUMAN_ONLY and MODEL_ONLY to True"
    assert not (TEST_HUMAN_MESSAGE and TEST_MODEL_MESSAGE), "Cannot set both TEST_HUMAN_MESSAGE and TEST_MODEL_MESSAGE to True"
    assert not (human_first_turn and MODEL_ONLY), "Cannot set human_first_turn and MODEL_ONLY to True"

    print(f"Setting human_first_turn to {human_first_turn}")
    print(f"Setting HUMAN_ONLY to {HUMAN_ONLY}")
    print(f"Setting MODEL_ONLY to {MODEL_ONLY}")
    print(f"Setting TEST_HUMAN_MESSAGE to {TEST_HUMAN_MESSAGE}")
    print(f"Setting TEST_MODEL_MESSAGE to {TEST_MODEL_MESSAGE}")

    # Load dataset
    lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train")
    turns = list(lmsys["turn"])

    # Filter dataset
    valid_indices = [i for i in range(len(turns)) if turns[i] == max_turns]
    conversations = lmsys[valid_indices[:1000]]
    num_conversations = len(conversations["conversation"])

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    results = []
    for i in range(num_conversations):

        if sum([len(message["content"]) for message in conversations["conversation"][i]]) > 100000:
            # Skip overly long conversations due to GPU constraints
            continue

        conv_results = dict()
        conv_results["conversation_id"] = conversations["conversation_id"][i]
        conversation = conversations["conversation"][i]

        if LOW_TOKENS:
            content = "".join([message["content"] for message in conversation])
            if len(tokenizer(content)["input_ids"]) > 4000:
                continue

        # Model answer is always last element
        test_index = -2 if TEST_HUMAN_MESSAGE else -1
        test_message = conversation[test_index]

        if TEST_HUMAN_MESSAGE:
            assert test_message["role"] == "user"
            if INJECT_RANDOM_TOPIC:
                content = "That's great to hear! Can you now give me a recipe for cooking a carbonara, true Italian style?"
            else:
                content = test_message["content"]

            final_formatted_message = "<|im_start|>user\n" + content + "<|im_end|>\n"

        elif TEST_MODEL_MESSAGE:
            final_model_input = conversation[-1]
            if INJECT_RANDOM_TOPIC:
                content = (
                    "To make authentic Carbonara, first brown 100g of sliced guanciale in a pan until crispy, "
                    "then set the pan aside. Whisk 3 egg yolks and 1 whole egg with 50g of finely grated "
                    "Pecorino Romano and plenty of freshly cracked black pepper to form a thick paste. "
                    "Boil 200g of pasta in salted water until al dente, reserving a small cup of the pasta "
                    "water before draining. Toss the hot pasta into the pan with the guanciale fat, then—with "
                    "the heat strictly turned off—pour in the egg mixture and a splash of pasta water. Stir "
                    "vigorously and continuously until the residual heat creates a glossy, creamy emulsion. "
                    "Serve immediately with an extra dusting of cheese and pepper."
                )
            else:
                content = final_model_input["content"]

            final_formatted_message = "<|im_start|>assistant\n" + content + "<|im_end|>\n"

        conversation = conversation[:test_index]

        if MIX_INPUTS:
            assert HUMAN_ONLY, "Mixing inputs only supported for HUMAN_ONLY setting"
            random.shuffle(conversation)

        # Don't care about padding, attn_mask since no batch processing
        output_ids = tokenizer(final_formatted_message, return_tensors="pt")
        conv_results["test_message_length"] = len(output_ids["input_ids"][0])

        for num_turns in range(0, max_turns):

            # Use last num_turns turns of conversation as conditioning
            if num_turns == 0:
                conversation_subset = []
            else:
                conversation_subset = (
                    conversation[-num_turns * 2:]
                    if TEST_HUMAN_MESSAGE
                    else conversation[(-num_turns * 2 - 1):]
                )

            if HUMAN_ONLY or MODEL_ONLY:
                role = "user" if HUMAN_ONLY else "assistant"
                conversation_subset = [msg for msg in conversation_subset if msg["role"] == role]

            if INVESTIGATOR_SETTING:
                assert TEST_HUMAN_MESSAGE, "Investigator prompt is specifically designed to estimate user input."
                system_instruction = {"content": INVESTIGATOR_PROMPT, "role": "system"}
            else:
                system_instruction = {"content": "You are a helpful assistant.", "role": "system"}

            if human_first_turn and len(conversation_subset) > 0 and conversation_subset[0]["role"] == "assistant":
                conversation_subset = conversation_subset[1:]
                assert conversation_subset[0]["role"] == "user"

            conversation_subset = [system_instruction] + conversation_subset
            input_ids = apply_template(conversation_subset, tokenizer)

            complete_sequence = torch.cat((input_ids["input_ids"], output_ids["input_ids"]), dim=-1).to(device)

            # Get model logits
            with torch.no_grad():
                logits = model(complete_sequence).logits

            # Compute logprobs for output tokens
            output_logits = logits[:, -output_ids["input_ids"].shape[1] - 1 : -1, :]
            probs = torch.nn.functional.softmax(output_logits, dim=-1)
            log_probs = torch.log(probs)

            target_tokens = output_ids["input_ids"].to(device)
            assert target_tokens.shape[1] == log_probs.shape[1]

            cum_logprob = 0.0
            absolute_rankings = []
            for k in range(target_tokens.shape[1]):
                token_id = target_tokens[0, k].item()
                logprob = log_probs[0, k, token_id].item()
                cum_logprob += logprob
                absolute_rankings.append(
                    log_probs[0, k].argsort(descending=True).tolist().index(token_id)
                )

            entropy = -torch.sum(probs * torch.log(probs), dim=-1).sum().item()

            conv_results[f"logprob_turns_{num_turns + 1}"] = cum_logprob
            conv_results[f"entropy_turns_{num_turns + 1}"] = entropy
            conv_results[f"mean_rank_turns_{num_turns + 1}"] = np.mean(absolute_rankings)
            conv_results[f"median_rank_turns_{num_turns + 1}"] = np.median(absolute_rankings)

        results.append(conv_results)

    # Save results
    print(f"Recorded {len(results)} results")
    input_setting = "HO" if HUMAN_ONLY else "MO" if MODEL_ONLY else "all"
    output_dir = f"results/exp_multi_turn/{input_setting}"
    os.makedirs(output_dir, exist_ok=True)

    output_setting = "TH" if TEST_HUMAN_MESSAGE else "TM"
    inject_message = "_RT" if INJECT_RANDOM_TOPIC else ""
    investigator_setting = "_INV" if INVESTIGATOR_SETTING else ""
    mix_inputs_setting = "_MIX" if MIX_INPUTS else ""
    low_token_setting = "_LT" if LOW_TOKENS else ""
    output_path = (
        f"{output_dir}/exp_multi_{input_setting}_{output_setting}"
        f"{inject_message}{investigator_setting}{mix_inputs_setting}"
        f"{low_token_setting}_{max_turns}_turn.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    end = time.time()
    print(f"Experiment completed in {end - start:.2f} seconds.")
    print("=" * 20)


if __name__ == "__main__":

    # Experiment flags
    INJECT_RANDOM_TOPIC = False  # Negative control: replace target with off-topic message
    INVESTIGATOR_SETTING = False  # Replace system prompt with investigator prompt
    MIX_INPUTS = False  # Randomly permute user message order
    HUMAN_ONLY = False  # Context includes only user messages
    MODEL_ONLY = False  # Context includes only assistant messages
    TEST_HUMAN_MESSAGE = True  # Score the last user message
    TEST_MODEL_MESSAGE = False  # Score the last assistant response
    LOW_TOKENS = True  # Filter out conversations exceeding 4000 tokens

    for max_turns in [5, 10, 20]:
        main(max_turns=max_turns)