# Current Findings

Short review after reading `rag_system.py`, `run_attribution_experiments.py`, `attribution_methods.py`, and `data_loader.py`.

## Architecture Walkthrough

- `RAGSystem` loads HF causal LM + tokenizer, formats prompts (`Context + Question -> Answer`), generates responses, and scores subsets via log-prob of a provided target response.
- `run_attribution_experiments.py` loads a dataset, builds target responses (uses gold answer when present), then for each query runs multiple attribution methods from `attribution_methods.py` and logs simple metrics (top-2 accuracy, ranks).
- Attribution methods include leave-one-out and several Shapley approximations; all call a passed `utility_function` that in turn calls `RAGSystem.compute_utility`.

## Findings / Risks

- Target token masking includes the final prompt token, so the utility sums one extra prompt token log-probability. The mask currently starts at `prompt_length-1`; it should start at `prompt_length` to exclude the prompt itself. This can inflate utilities and distort attribution comparisons.

```
269:284:rag_system.py
        # Mask to target tokens only (positions at/after first target token)
        target_mask = torch.zeros_like(labels, dtype=torch.bool)
        target_mask[:, prompt_length-1:] = True
        effective_mask = target_mask & (attn_shifted > 0)
```

- Utility evaluation is very expensive: every attribution call runs a full forward pass over `prompt + target`. For Shapley/permutation loops (dozens of calls per query) this leads to slow runs and likely explains the need to interrupt runs (`KeyboardInterrupt` in the terminal trace). Consider caching the model outputs for the full prompt or using prefix-caching/kv-caching to reuse prompt computations.
- No sequence-length guardrails: prompts concatenate all documents with no truncation to the model’s max length. Long contexts will silently truncate in the tokenizer or produce poor scores; add explicit max length and truncation strategy (e.g., per-doc clipping or sliding window).
- Metrics assume relevant docs are always `['A', 'B']`, use absolute attributions, and treat any non-null rank as success. On datasets with different gold doc sets or when polarity matters, these metrics may mislead.
- Dependency pinning drift: `pyproject.toml` now declares `numpy>=2.1.3` and Python `>=3.10`, but the lockfile still pulls `numpy==1.26.4` and failed to build during `uv sync`. Regenerate the lock (`uv lock --upgrade`) to align with the declared floor and avoid source builds on macOS/3.13.

## Quick Suggestions

- Fix the target-mask start index in `compute_log_probability`.
- Add prompt-length control (max tokens, truncation policy) and consider batching utilities or caching logits for repeated calls.
- Re-lock dependencies to match the updated `pyproject.toml` and ensure prebuilt wheels are available for the target Python version.
