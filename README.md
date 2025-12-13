# RAG Source Attribution

This repository implements and evaluates source attribution methods for Retrieval-Augmented Generation (RAG) systems. It provides both black-box (Shapley-based) and white-box (gradient/attention-based) attribution techniques to determine which retrieved documents contribute most to a generated response, enabling trustworthiness analysis of RAG outputs.

## Setup

```bash
pip install -e .   # or: uv sync
```

**HuggingFace Token:** Set `HF_TOKEN` or `HUGGINGFACE_TOKEN` environment variable, or pass `token=` parameter.

## Running Experiments

### Black-box Attribution (Shapley-based)

```bash
python run_attribution_experiments.py --dataset data/complementary.csv --methods permutation_shapley monte_carlo_shapley --max-queries 10
```

### White-box Attribution (Gradient/Attention-based)

```bash
python run_whitebox_attribution_experiments.py --dataset data/complementary.csv --methods gradient attention --max-queries 10
```

### Programmatic Usage

**Black-box (Shapley-based):**

```python
from run_attribution_experiments import run_attribution_experiment, save_results

results = run_attribution_experiment(
    dataset_path="data/complementary.csv",
    model_name="meta-llama/Llama-3.2-1B",
    device="cuda",
    max_queries=10,
    methods=["permutation_shapley", "monte_carlo_shapley"]
)
save_results(results, output_dir="results")
```

**White-box (Gradient/Attention-based):**

```python
from run_whitebox_attribution_experiments import run_whitebox_attribution_experiment, save_results

results = run_whitebox_attribution_experiment(
    dataset_path="data/complementary.csv",
    model_name="meta-llama/Llama-3.2-1B",
    device="cuda",
    max_queries=10,
    methods=["gradient", "attention"]
)
save_results(results, output_dir="results")
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Path to dataset CSV file |
| `--model` | HuggingFace model name (default: `meta-llama/Llama-3.2-1B`) |
| `--device` | `cuda` or `cpu` (auto-detect if omitted) |
| `--max-queries` | Limit number of queries to process (useful for testing) |
| `--methods` | Attribution methods to run (see below) |
| `--output-dir` | Output directory for results (default: `results/`) |

### Datasets

- **complementary** – Each question requires combining info from exactly 2 documents (A, B); other docs are distractors
- **synergy** – Answer requires synthesizing information across 2 documents; neither alone suffices
- **duplicate** – Documents A and B contain redundant/equivalent information for the answer

### Methods

**Black-box (`run_attribution_experiments.py`):**
- `leave_one_out` – Measures utility drop when each document is removed
- `permutation_shapley` – Approximates Shapley values via random permutations
- `monte_carlo_shapley` – Monte Carlo sampling of coalition values
- `exact_shapley` – Exact Shapley computation (≤5 docs only)

**White-box (`run_whitebox_attribution_experiments.py`):**
- `gradient` – Gradient of utility w.r.t. input embeddings
- `integrated_gradients` – Path-integral gradient method (requires `captum`)
- `attention` – Aggregated attention weights over document tokens

## Output

Results are saved to `results/` as:
- `{dataset}_{timestamp}_full.json` – Full per-query results
- `{dataset}_{timestamp}_metrics.csv` – Aggregate metrics (top-2 accuracy, mean ranks)

## Visualizations

```bash
python create_visualizations.py summary              # Multi-run summary from results/
python create_visualizations.py run --input results/file_full.json   # Per-run figures
python create_visualizations.py all                  # All figures
python create_final_visualizations.py               # Paper-ready figures → figures/final/
```

Figures are saved to `figures/`.
