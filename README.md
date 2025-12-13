# RAG Source Attribution

Source attribution for Retrieval-Augmented Generation (RAG) systems using cooperative game theory (Shapley values) and baseline methods.

Based on the methodology from [arXiv:2507.04480](https://arxiv.org/abs/2507.04480).

## Overview

This project implements and evaluates attribution methods for identifying which source documents in a RAG system contribute most to the generated response. The system uses:

- **Shapley Value-based Methods**: Exact, Monte Carlo, Permutation-based, and Kernel SHAP approximations
- **Baseline Methods**: Leave-One-Out, Gradient-based, Attention-based, Retrieval Score
- **Evaluation Metrics**: Top-2 Accuracy and Mean Rank of gold documents

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for faster inference)
- Hugging Face account and access token (for downloading models)

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd RAG-Attribution
   ```

2. **Install dependencies using `uv` (recommended):**

   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync
   ```

   Alternatively, use `pip`:

   ```bash
   pip install -e .
   ```

3. **Set up Hugging Face token:**

   ```bash
   # Option 1: Set environment variable
   export HF_TOKEN="your_token_here"

   # Option 2: Create .env file
   echo "HF_TOKEN=your_token_here" > .env

   # Option 3: Login via Hugging Face CLI
   huggingface-cli login
   ```

   To retrieve your token if forgotten:

   ```bash
   cat ~/.cache/huggingface/token
   ```

## Usage

### Running Attribution Experiments

**Basic usage:**

```bash
uv run python run_attribution_experiments.py \
    --dataset data/20_complementary.csv \
    --model meta-llama/Llama-3.2-1B \
    --max-queries 20
```

**Full options:**

```bash
uv run python run_attribution_experiments.py \
    --dataset data/20_complementary.csv \
    --model meta-llama/Llama-3.2-1B \
    --device cuda \
    --max-queries 20 \
    --methods leave_one_out permutation_shapley monte_carlo_shapley
```

**Available methods:**

- `leave_one_out`
- `permutation_shapley`
- `monte_carlo_shapley`
- `kernel_shap`
- `gradient_attribution` (requires whitebox access)
- `integrated_gradients_attribution` (requires whitebox access)
- `attention_attribution` (requires whitebox access)

**Output:**

- Results saved to `results/{dataset_name}_{timestamp}_full.json`
- Metrics saved to `results/{dataset_name}_{timestamp}_metrics.csv`

### Running Whitebox Attribution Experiments

For gradient-based and attention-based methods:

```bash
uv run python run_whitebox_attribution_experiments.py \
    --dataset data/20_complementary.csv \
    --model meta-llama/Llama-3.2-1B \
    --max-queries 20
```

**Note:** Whitebox methods require `attn_implementation='eager'` and may have limited performance.

### Generating Visualizations

**All visualizations:**

```bash
uv run python create_visualizations.py all \
    --results-dir results \
    --output-dir figures
```

**Individual visualization types:**

1. **Summary visualizations** (includes ablation study):

   ```bash
   uv run python create_visualizations.py summary \
       --results-dir results \
       --output-dir figures
   ```

2. **Per-run visualizations** (for a single experiment):

   ```bash
   uv run python create_visualizations.py run \
       --input results/20_complementary_20251212_221510_full.json \
       --output-dir figures
   ```

3. **Extra report figures:**
   ```bash
   uv run python create_visualizations.py extras \
       --results-dir results \
       --output-dir figures
   ```

**Using specific result files:**

```bash
uv run python create_visualizations.py summary \
    --files results/20_complementary_20251212_221510_full.json \
           results/20_duplicate_20251212_221425_full.json \
           results/20_synergy_20251212_222327_full.json \
    --output-dir figures
```

## Datasets

The project includes synthetic datasets designed to test different attribution scenarios:

- **`20_complementary.csv`**: Documents A and B together provide the complete answer
- **`20_synergy.csv`**: Documents interact synergistically
- **`20_duplicate.csv`**: Documents contain redundant information
- **`inverse_synergy.csv`**: Documents may interfere with each other

Each dataset contains 20 queries with 10 documents each, where documents A and B are the "gold" documents that should be attributed highest importance.

## Reproducing Results

### Step 1: Run Experiments

Run experiments on all datasets:

```bash
# Complementary dataset
uv run python run_attribution_experiments.py \
    --dataset data/20_complementary.csv \
    --model meta-llama/Llama-3.2-1B \
    --max-queries 20

# Duplicate dataset
uv run python run_attribution_experiments.py \
    --dataset data/20_duplicate.csv \
    --model meta-llama/Llama-3.2-1B \
    --max-queries 20

# Synergy dataset
uv run python run_attribution_experiments.py \
    --dataset data/20_synergy.csv \
    --model meta-llama/Llama-3.2-1B \
    --max-queries 20
```

### Step 2: Generate Visualizations

After running experiments, generate all visualizations:

```bash
uv run python create_visualizations.py all \
    --results-dir results \
    --output-dir figures
```

This will create:

- `figures/architecture.png` - System architecture diagram
- `figures/method_comparison.png` - Method performance comparison
- `figures/ablation_study.png` - Ablation study (method performance + ranking method impact)
- `figures/dataset_method_summary_top2.png` - Top-2 accuracy by dataset and method
- `figures/dataset_method_summary_rank.png` - Mean rank by dataset and method
- `figures/attribution_example.png` - Example attribution scores
- `figures/per_query_analysis.png` - Per-query performance analysis
- `figures/summary_table.png` - Summary table of metrics
- `figures/trustworthy_aspects.png` - Trustworthiness analysis
- `figures/baseline_comparison.png` - Baseline methods comparison
- `figures/hyperparameter_sensitivity.png` - Hyperparameter sensitivity (ranking method uses real data)
- `figures/challenges.png` - Challenges and limitations

### Step 3: Verify Results

Check the generated metrics CSV files:

```bash
# View combined metrics
cat results/combined_metrics_*.csv
```

## Hyperparameter Sensitivity Analysis

The hyperparameter sensitivity plot (`figures/hyperparameter_sensitivity.png`) includes:

1. **Monte Carlo Shapley: Sample Size Sensitivity** (Top-left)

   - **Status**: Illustrative placeholder values
   - **To calculate**: Run experiments with different `num_samples` values (16, 32, 64, 128, 256)
   - **Current setting**: `num_samples=64` (hardcoded in `run_attribution_experiments.py`)

2. **Permutation Shapley: Permutation Count Sensitivity** (Top-right)

   - **Status**: Illustrative placeholder values
   - **To calculate**: Run experiments with different `num_permutations` values (10, 25, 50, 100, 200)
   - **Current setting**: `num_permutations=50` (hardcoded in `attribution_methods.py`)

3. **Target Response Generation: Token Count Sensitivity** (Bottom-left)

   - **Status**: Illustrative placeholder values
   - **To calculate**: Run experiments with different `max_new_tokens` values (25, 50, 75, 100)
   - **Current setting**: Uses model default or `max_new_tokens` parameter in `RAGSystem`

4. **Ranking Method Impact** (Bottom-right)
   - **Status**: ✅ **Uses real data from your experiments**
   - **Calculated from**: All `*_full.json` files in `results/` directory
   - **Compares**: Raw score ranking vs. absolute value ranking

To generate real hyperparameter sensitivity data, modify the experiment scripts to:

- Accept hyperparameter values as command-line arguments
- Store hyperparameter values in the results JSON
- Run multiple experiments with different hyperparameter values
- Aggregate results by hyperparameter value

## Project Structure

```
RAG-Attribution/
├── rag_system.py              # Core RAG system and utility computation
├── attribution_methods.py     # All attribution algorithms
├── run_attribution_experiments.py    # Main experiment runner
├── run_whitebox_attribution_experiments.py  # Whitebox methods runner
├── create_visualizations.py  # Visualization CLI entry point
├── visualizations/            # Visualization modules
│   ├── summary.py            # Multi-run summary visualizations
│   ├── per_run.py            # Single-run visualizations
│   ├── extras.py             # Additional report figures
│   └── common.py             # Shared utilities
├── data/                      # Dataset files
├── results/                   # Experiment results (JSON and CSV)
├── figures/                   # Generated visualization files
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Key Implementation Details

### Log Probability Calculation

The core utility function `v(S) = log P(R_target | Q, S)` is computed using causal language model log probabilities:

- Logits and labels are shifted by one position (logits predict next token)
- Attention masks are applied correctly
- Only target tokens are included in the sum (not prompt tokens)

### Attribution Methods

- **Exact Shapley**: Computes exact Shapley values (only feasible for ≤10 documents)
- **Monte Carlo Shapley**: Samples random subsets, averages marginal contributions
- **Permutation Shapley**: Samples random permutations, averages marginal contributions
- **Kernel SHAP**: Uses SHAP library's kernel explainer
- **Leave-One-Out**: Computes v(D) - v(D\{d_i}) for each document

### Evaluation Metrics

- **Top-2 Accuracy**: Whether the top-2 documents (by absolute attribution score) match gold documents A and B
- **Mean Rank**: Average rank of gold documents in the attribution ranking (lower is better, ideal = 1.5)

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure you're using `uv run`:

```bash
uv run python <script>.py
```

### CUDA Out of Memory

Reduce batch size or use CPU:

```bash
uv run python run_attribution_experiments.py --device cpu ...
```

### Hugging Face Authentication

If model download fails:

1. Check your token: `cat ~/.cache/huggingface/token`
2. Set environment variable: `export HF_TOKEN="your_token"`
3. Or login: `huggingface-cli login`

## Citation

If you use this code, please cite the original paper:

```
@article{rag_attribution_2025,
  title={RAG Source Attribution via Cooperative Game Theory},
  author={...},
  journal={arXiv preprint arXiv:2507.04480},
  year={2025}
}
```

## License

[Add your license here]

## Contact

[Add contact information here]
