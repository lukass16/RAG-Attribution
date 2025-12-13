# RAG Source Attribution Analysis

## Project Overview

This project implements and evaluates methods for **source attribution in Retrieval-Augmented Generation (RAG) systems**. Given a question and a set of retrieved documents, the system identifies which documents are most influential in generating the final answer. This is critical for explainability, trustworthiness, and debugging RAG systems.

The work is based on the theoretical framework from [arXiv:2507.04480](https://arxiv.org/abs/2507.04480), which applies cooperative game theory (specifically Shapley values) to attribute contributions of individual documents to the RAG system's output.

## Problem Statement

In RAG systems, multiple documents are retrieved and used as context to answer a question. However, not all documents contribute equally to the final answer. Some may be:

- **Essential**: Directly contain information needed to answer the question
- **Complementary**: Provide additional context that enhances the answer
- **Redundant**: Duplicate information already present in other documents
- **Irrelevant**: Don't contribute to answering the question

**Source attribution** answers: "Which documents were most important for generating this answer?"

## Methodology

### Core Framework

The attribution problem is framed as a **cooperative game**:

1. **Players**: Documents D = {d₁, d₂, ..., dₙ}
2. **Utility Function**: v(S) = log P(R_target | Q, S)
   - Measures how well a document subset S supports generating the target response R_target
   - Computed as the log probability of generating the target response given the question Q and subset S
3. **Attribution Scores**: φᵢ for each document dᵢ
   - Quantifies the marginal contribution of document i across all possible document coalitions

### Attribution Methods Implemented

#### 1. **Shapley Value Methods**

**Exact Shapley Values**

- Computes the true Shapley value using the formula:
  ```
  φᵢ = Σ_{S⊆D\{dᵢ}} [|S|!(|D|-|S|-1)!/|D|!] * [v(S∪{dᵢ}) - v(S)]
  ```
- Requires 2ⁿ utility evaluations (exponential complexity)
- Only feasible for small document sets (n ≤ 5)

**Monte Carlo Shapley**

- Approximates Shapley values by sampling random permutations
- For each permutation, computes marginal contributions
- Averages over multiple samples (default: 64-100 samples)
- Linear complexity: O(n × samples)

**Permutation Shapley**

- Similar to Monte Carlo but explicitly samples permutations
- Default: 50 permutations
- More systematic sampling approach

**Kernel SHAP**

- Uses weighted linear regression to approximate Shapley values
- Leverages the SHAP library's KernelExplainer
- Efficient for medium-sized document sets

#### 2. **Baseline Methods**

**Leave-One-Out (LOO)**

- Simple heuristic: φᵢ = v(D) - v(D\{dᵢ})
- Measures the drop in utility when document i is removed
- Very fast: O(n) utility evaluations
- Doesn't account for interactions between documents

**Gradient-Based Attribution** (optional, requires Captum)

- Computes gradients of utility w.r.t. token embeddings
- Aggregates gradients to document level
- Provides token-level insights

**Attention-Based Attribution** (optional)

- Uses model attention weights to attribute importance
- Directly leverages the model's internal representations

## Implementation Details

### Architecture

The system consists of four main components:

1. **Input**: Question Q and document set D = {d₁, ..., dₙ}
2. **Generation**: RAG system (LLM) generates target response R_target using all documents
3. **Evaluation**: Utility function v(S) computed for various document subsets S ⊆ D
4. **Attribution**: Attribution methods compute scores φᵢ for each document

### Key Technical Components

#### RAG System (`rag_system.py`)

**RAGSystem Class**

- Wraps a HuggingFace language model (default: `meta-llama/Llama-3.2-1B`)
- Handles tokenization, generation, and log probability computation
- Supports configurable attention implementations (`eager`, `sdpa`)
- Implements input/document truncation for long contexts

**Key Methods**:

- `generate_response()`: Generate answer given question and document subset
- `generate_target_response()`: Generate gold-standard response using all documents
- `compute_utility()`: Compute v(S) = log P(R_target | Q, S)
- `compute_log_probability()`: Core utility computation using causal LM log probabilities

**Critical Fix**: The log probability computation was corrected to properly handle causal language models:

- Shifts logits and labels by one position (logits predict next token)
- Applies attention masks correctly
- Only sums log probabilities over target tokens (not prompt tokens)

#### Attribution Methods (`attribution_methods.py`)

Implements all attribution algorithms as functions that take:

- `documents`: List of document strings
- `utility_function`: Callable that computes v(S) for a subset S

Returns a dictionary mapping document index to attribution score.

#### Experiment Runner (`run_attribution_experiments.py`)

**Main Function**: `run_attribution_experiment()`

- Loads dataset from CSV/JSON
- Generates target responses (prefers gold answers from dataset when available)
- Computes attributions for each method
- Evaluates performance using metrics
- Saves results to JSON and CSV files

**Evaluation Metrics**:

- **Top-2 Accuracy**: Whether the two most important documents (by absolute attribution score) match the expected gold documents
- **Mean Rank**: Average rank of gold documents in the attribution ranking (lower is better, ideal = 1.5)

### Datasets

The project uses synthetic datasets designed to test different attribution scenarios:

#### Dataset Types

1. **Complementary** (`complementary.csv`, `20_complementary.csv`)

   - Documents A and B together provide the complete answer
   - Each document alone is insufficient
   - Tests whether methods can identify both essential documents

2. **Synergy** (`20_synergy.csv`)

   - Documents interact synergistically
   - Combined effect is greater than sum of parts
   - Challenges attribution methods to capture interactions

3. **Duplicate** (`20_duplicate.csv`)

   - Documents contain redundant information
   - Tests whether methods can identify redundancy
   - May assign lower scores to duplicates

4. **Inverse Synergy** (`inverse_synergy.csv`)
   - Documents may interfere with each other
   - Tests negative interactions

#### Dataset Format

CSV files with columns:

- `question`: The question to answer
- `context`: List of document strings (JSON array or Python list)
- `answer`: Gold-standard answer (used as target response)
- `id`: List of document IDs (typically ['A', 'B', 'C', ...])

Each dataset contains 10-20 queries, each with 10 documents where documents A and B are the "gold" documents that should be attributed highest importance.

## Results

### Performance Summary

Based on experiments run on December 12, 2025 (`combined_metrics_20251212_221425.csv`):

#### Complementary Dataset (20 queries)

- **Leave-One-Out**: Top-2 accuracy = 1.0, Mean ranks: A=1.65, B=1.35
- **Permutation Shapley**: Top-2 accuracy = 1.0, Mean ranks: A=1.75, B=1.25
- **Monte Carlo Shapley**: Top-2 accuracy = 0.95, Mean ranks: A=1.8, B=1.25

#### Synergy Dataset (20 queries)

- **Leave-One-Out**: Top-2 accuracy = 0.7, Mean ranks: A=3.2, B=1.1
- **Permutation Shapley**: Top-2 accuracy = 0.25, Mean ranks: A=4.0, B=1.15
- **Monte Carlo Shapley**: Top-2 accuracy = 0.3, Mean ranks: A=3.55, B=1.05

#### Duplicate Dataset (20 queries)

- **Leave-One-Out**: Top-2 accuracy = 0.7, Mean ranks: A=1.15, B=3.1
- **Permutation Shapley**: Top-2 accuracy = 0.95, Mean ranks: A=1.1, B=1.95
- **Monte Carlo Shapley**: Top-2 accuracy = 0.9, Mean ranks: A=1.15, B=2.0

### Key Findings

1. **Leave-One-Out performs best on complementary scenarios** (perfect 1.0 accuracy)
2. **Shapley methods struggle with synergy** (lower accuracy, higher ranks for document A)
3. **Permutation Shapley excels on duplicate scenarios** (0.95 accuracy)
4. **Ranking by absolute value is crucial** - raw scores can be negative, but magnitude indicates importance

## Technical Improvements Made

### 1. Fixed Log Probability Computation

**Problem**: Original implementation incorrectly computed log probabilities for causal LMs:

- Used logits at the same position as the token (should predict next token)
- Didn't properly handle attention masks
- Included prompt tokens in probability calculation

**Solution**:

- Shift logits and labels by one position (causal LM alignment)
- Apply attention masks correctly
- Mask to only sum probabilities over target tokens

### 2. Use Gold Answers as Target Responses

**Problem**: Using model-generated responses as targets introduced noise and inconsistency

**Solution**: Prefer dataset-provided gold answers when available, falling back to model generation only if missing

### 3. Added Document Truncation

**Problem**: Long documents could exceed model context limits

**Solution**:

- Configurable `max_input_tokens` and `max_doc_tokens` parameters
- Automatic truncation with proper handling in utility computation

### 4. Enhanced Evaluation

**Improvements**:

- Support for custom gold document sets (not just ['A', 'B'])
- Timing information for each method
- Absolute value attribution scores stored separately
- More detailed result JSONs with raw and processed documents

## Usage

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Setting Up HuggingFace Token

Create a `.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
```

Or set environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

### Running Experiments

```bash
# Run on a dataset with default methods
python run_attribution_experiments.py \
    --dataset data/20_complementary.csv \
    --max-queries 5

# Specify methods explicitly
python run_attribution_experiments.py \
    --dataset data/20_synergy.csv \
    --methods leave_one_out permutation_shapley monte_carlo_shapley \
    --max-queries 10

# Use a different model
python run_attribution_experiments.py \
    --dataset data/complementary.csv \
    --model meta-llama/Llama-3.1-8B
```

### Generating Visualizations

```bash
# Summary visualizations (multi-run)
python create_visualizations.py summary \
    --results-dir results \
    --output-dir figures

# Per-run visualizations
python create_visualizations.py run \
    --input results/20_complementary_20251212_221510_full.json \
    --output-dir figures

# Additional report visualizations
python create_visualizations.py extras \
    --results-dir results \
    --output-dir figures

# All visualizations
python create_visualizations.py all \
    --results-dir results \
    --output-dir figures
```

## Project Structure

```
RAG-Attribution/
├── rag_system.py              # Core RAG system and utility computation
├── attribution_methods.py     # All attribution algorithms
├── run_attribution_experiments.py  # Main experiment runner
├── create_visualizations.py   # Unified visualization CLI
├── visualizations/            # Visualization modules
│   ├── common.py              # Shared utilities
│   ├── summary.py             # Multi-run summary plots
│   ├── per_run.py             # Single-run analysis
│   └── extras.py              # Additional report figures
├── data/                      # Datasets
│   ├── complementary.csv
│   ├── 20_complementary.csv
│   ├── 20_synergy.csv
│   ├── 20_duplicate.csv
│   └── ...
├── results/                   # Experiment results
│   ├── *_full.json            # Complete results per run
│   ├── *_metrics.csv          # Aggregated metrics per run
│   └── combined_metrics_*.csv # Combined metrics across runs
├── figures/                   # Generated visualizations
│   ├── architecture.png
│   ├── method_comparison.png
│   ├── summary_table.png
│   └── ...
├── pyproject.toml             # Project dependencies
└── PROJECT_DESCRIPTION.md     # This file
```

## Output Files

### Result JSON (`*_full.json`)

Contains:

- Dataset and model information
- Per-query results with:
  - Question, target response, documents
  - Attribution scores for each method
  - Top-2 accuracy and ranks
  - Timing information
- Aggregate metrics across all queries

### Metrics CSV (`*_metrics.csv`)

Aggregated metrics per method:

- `method`: Attribution method name
- `top2_accuracy`: Fraction of queries where top-2 documents are correct
- `mean_rank_A`, `mean_rank_B`: Average rank of gold documents
- `n_queries`: Number of queries evaluated

### Visualizations

- **Architecture diagram**: System overview
- **Method comparison**: Performance across methods
- **Dataset summary**: Performance by dataset type
- **Per-query analysis**: Detailed breakdown
- **Attribution examples**: Sample attribution scores
- **Ablation studies**: Hyperparameter sensitivity
- **Trustworthy aspects**: Explainability and reliability metrics

## Future Work

1. **Better handling of synergy scenarios**: Current methods struggle when documents interact non-linearly
2. **Efficient approximations**: Faster Shapley approximations for large document sets
3. **Token-level attribution**: Extend to identify which parts of documents are most important
4. **Real-world datasets**: Test on actual RAG applications (e.g., question answering over Wikipedia)
5. **Comparative evaluation**: Compare against attention-based and gradient-based methods more systematically

## References

- Original paper: [arXiv:2507.04480](https://arxiv.org/abs/2507.04480) - "Source Attribution in RAG Systems"
- Shapley values: Shapley, L. S. (1953). A value for n-person games.
- SHAP library: Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.

## License

[Specify your license here]

## Authors

[Your name/institution]

