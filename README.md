# Investigating the Predictive Role of User Information in Large Language Models Through Dialogue

**Author:** Allegra Boustany, École Polytechnique  
**Advisor:** Professor Luca Biggio, Bocconi University  
**Academic Year:** 2025/2026

## Overview

This repository contains the main scripts used for the experiments in the thesis. The code is organized by thesis section.

## Structure

```
section4_real_conversations/    Section 4.2: Real Multi-Turn Conversations
section4_synthetic_personas/    Section 4.3: Controlled Synthetic Conversations
section5_special_tokens/        Section 5: Compressing Dialogue-Derived User Information into Learned Special Tokens
```

## Models Used

| Model | Role | Section |
|-------|------|---------|
| Qwen3-8B | Evaluator/scorer for real conversations | 4.2 |
| Qwen3-4B-Instruct-2507 | Generation and scoring for synthetic conversations | 4.3 |
| Qwen3-4B-Instruct-2507 | Conversation generation for special token dataset | 5.2 |
| Qwen2.5-0.5B | Special token training and evaluation | 5.3 |

## Dependencies

- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- datasets (HuggingFace)
- numpy
- PyYAML (for Section 4 config files)

## Datasets

- **LMSYS-Chat-1M** (Section 4.2): Real user-assistant conversations. Available on HuggingFace (`lmsys/lmsys-chat-1m`).
- **PersonaChat** (Section 4.3): Source of persona descriptions for hidden-persona condition. Available on HuggingFace (`AlekseyKorshuk/persona-chat`).
- **UltraChat** (Section 4.3): Source of seed prompts for synthetic conversations. Available on HuggingFace (`HuggingFaceH4/ultrachat_200k`).
- **Custom persona dataset** (Section 5): Manually defined personas with controlled attributes. See `section4_special_tokens/config/`.
