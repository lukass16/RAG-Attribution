# Visualization Summary

This document describes all the visualizations created for the RAG Source Attribution project.

## Generated Figures

### 1. Architecture Diagram (`architecture.png`)
- **Purpose**: Illustrates the overall system architecture
- **Shows**: 
  - Flow from Question + Documents → RAG System → Target Response
  - Utility function computation
  - Attribution methods application
  - Output of attribution scores
- **Use for**: Architecture Illustration (5% of grade)

### 2. Method Comparison (`method_comparison.png`)
- **Purpose**: Compare different attribution methods
- **Shows**:
  - Top-2 accuracy across methods
  - Mean rank of Document A
  - Mean rank of Document B
- **Use for**: Baselines (15%) and Results (40%)

### 3. Attribution Example (`attribution_example.png`)
- **Purpose**: Show detailed attribution scores for a single query
- **Shows**:
  - Attribution scores for each document (sorted by absolute value)
  - Highlights documents A and B in red
  - Comparison across multiple methods
- **Use for**: Results - Presentation (10%)

### 4. Per-Query Analysis (`per_query_analysis.png`)
- **Purpose**: Analyze performance across different queries
- **Shows**:
  - Top-2 accuracy variation across queries
  - Mean rank variation across queries
- **Use for**: Results - Trustworthy Aspects (15%)

### 5. Summary Table (`summary_table.png`)
- **Purpose**: Tabular summary of all methods
- **Shows**:
  - Top-2 accuracy
  - Mean ranks for A and B
  - Number of queries evaluated
- **Use for**: Results - Presentation (10%)

### 6. Ablation Study (`ablation_study.png`)
- **Purpose**: Show ablation studies
- **Shows**:
  - Sensitivity to sample size (Monte Carlo Shapley)
  - Impact of ranking by absolute value vs raw score
- **Use for**: Ablation Studies (15%)

### 7. Trustworthy Aspects (`trustworthy_aspects.png`)
- **Purpose**: Focus on trustworthy aspects
- **Shows**:
  - Explainability: Can we identify correct sources?
  - Reliability: Consistency of rankings
- **Use for**: Results - Trustworthy Aspects (15%)

### 8. Baseline Comparison (`baseline_comparison.png`)
- **Purpose**: Compare baseline methods
- **Shows**:
  - Comparison of Leave-One-Out and Permutation Shapley
  - Multiple metrics side-by-side
- **Use for**: Baselines (15%)

### 9. Hyperparameter Sensitivity (`hyperparameter_sensitivity.png`)
- **Purpose**: Ablation study on hyperparameters
- **Shows**:
  - Sample size sensitivity (Monte Carlo)
  - Permutation count sensitivity
  - Token count sensitivity
  - Ranking method impact
- **Use for**: Ablation Studies (15%)

### 10. Challenges and Limitations (`challenges.png`)
- **Purpose**: Visualize challenges
- **Shows**:
  - Computational complexity comparison
  - Accuracy vs efficiency tradeoff
- **Use for**: Challenges and Future Plans (20%)

## Key Findings Visualized

1. **Absolute Value Ranking**: Critical fix - ranking by absolute value correctly identifies A and B as most important
2. **Method Performance**: Leave-One-Out performs best with absolute value ranking
3. **Computational Tradeoffs**: Exact Shapley is intractable for large document sets
4. **Hyperparameter Sensitivity**: Methods are relatively robust to hyperparameter choices

## Usage in Report

- **Architecture**: Use `architecture.png` for Architecture Illustration section
- **Baselines**: Use `baseline_comparison.png` and `method_comparison.png`
- **Results**: Use `method_comparison.png`, `attribution_example.png`, `trustworthy_aspects.png`
- **Ablation**: Use `ablation_study.png` and `hyperparameter_sensitivity.png`
- **Challenges**: Use `challenges.png` for Challenges section

## Running the Scripts

```bash
# Generate all visualizations
python create_visualizations.py

# Generate additional specialized visualizations
python create_additional_visualizations.py
```

## Notes

- All figures are saved as high-resolution PNG files (300 DPI)
- Figures use consistent color schemes and styling
- Document A and B are highlighted in red in attribution plots
- All plots include proper labels, legends, and captions


