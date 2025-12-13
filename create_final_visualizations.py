#!/usr/bin/env python3
"""
Final visualization script for RAG Attribution paper.

Creates publication-ready figures combining whitebox and Shapley-based attribution results.
Outputs to figures/final/
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "font.family": "serif",
})

# Color palette - distinct colors for each method
COLORS = {
    # Shapley methods (blues/greens)
    "leave_one_out": "#2171b5",
    "permutation_shapley": "#6baed6",
    "monte_carlo_shapley": "#08519c",
    # Whitebox methods (oranges/reds)
    "gradient": "#fd8d3c",
    "integrated_gradients": "#e6550d",
    "attention": "#a63603",
    # Gold docs
    "gold": "#d62728",
    "other": "#7f7f7f",
}

METHOD_NAMES = {
    "leave_one_out": "Leave-One-Out",
    "permutation_shapley": "Permutation Shapley",
    "monte_carlo_shapley": "Monte Carlo Shapley",
    "gradient": "Gradient",
    "integrated_gradients": "Integrated Gradients",
    "attention": "Attention",
}

SHAPLEY_METHODS = ["leave_one_out", "permutation_shapley", "monte_carlo_shapley"]
WHITEBOX_METHODS = ["gradient", "integrated_gradients", "attention"]


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_results(shapley_files: List[str], whitebox_files: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Load Shapley and whitebox result files."""
    shapley_results = []
    whitebox_results = []
    
    for fp in shapley_files:
        if Path(fp).exists():
            shapley_results.append(load_json(fp))
        else:
            print(f"Warning: {fp} not found")
    
    for fp in whitebox_files:
        if Path(fp).exists():
            whitebox_results.append(load_json(fp))
        else:
            print(f"Warning: {fp} not found")
    
    return shapley_results, whitebox_results


def get_aggregate_metrics(results: List[Dict]) -> Dict[str, Dict]:
    """Extract aggregate metrics from result files."""
    metrics = {}
    
    for result in results:
        for method, method_metrics in result.get("aggregate_metrics", {}).items():
            if method not in metrics:
                metrics[method] = {
                    "top2_accuracy": [],
                    "mean_rank_A": [],
                    "mean_rank_B": [],
                    "n_queries": 0,
                }
            if method_metrics.get("top2_accuracy") is not None:
                metrics[method]["top2_accuracy"].append(method_metrics["top2_accuracy"])
            if method_metrics.get("mean_rank_A") is not None:
                metrics[method]["mean_rank_A"].append(method_metrics["mean_rank_A"])
            if method_metrics.get("mean_rank_B") is not None:
                metrics[method]["mean_rank_B"].append(method_metrics["mean_rank_B"])
            metrics[method]["n_queries"] += method_metrics.get("n_queries", 0)
    
    # Average the metrics
    for method in metrics:
        for key in ["top2_accuracy", "mean_rank_A", "mean_rank_B"]:
            values = metrics[method][key]
            metrics[method][key] = np.mean(values) if values else None
            metrics[method][f"{key}_std"] = np.std(values) if len(values) > 1 else 0
    
    return metrics


def get_timing_data(results: List[Dict]) -> Dict[str, List[float]]:
    """Extract timing data from results."""
    timings = {}
    
    for result in results:
        for query_result in result.get("results", []):
            for method, timing in query_result.get("timings_seconds", {}).items():
                if method not in timings:
                    timings[method] = []
                timings[method].append(timing)
    
    return timings


def find_best_query_result(results: List[Dict], methods: List[str]) -> Optional[Dict]:
    """Find a query result that has valid attributions for all specified methods."""
    for result in results:
        for qr in result.get("results", []):
            if "attributions" not in qr:
                continue
            
            valid = True
            for method in methods:
                attrs = qr["attributions"].get(method)
                if attrs is None or not isinstance(attrs, list) or len(attrs) == 0:
                    valid = False
                    break
                # Check for non-trivial attributions
                if all(a == 0 for a in attrs):
                    valid = False
                    break
            
            if valid:
                return qr
    
    return None


# ============================================================================
# PLOT 1: Attribution Example
# ============================================================================

def plot_attribution_example(
    shapley_results: List[Dict],
    whitebox_results: List[Dict],
    output_path: str
):
    """
    Create attribution example plot showing all 6 methods for a single query.
    2 rows x 3 columns layout.
    """
    ensure_dir(Path(output_path).parent)
    
    # Find a query that has valid data in both result sets
    # Use first query from whitebox as it has both
    shapley_query = find_best_query_result(shapley_results, SHAPLEY_METHODS)
    whitebox_query = find_best_query_result(whitebox_results, WHITEBOX_METHODS)
    
    if shapley_query is None or whitebox_query is None:
        print("Could not find valid query results for attribution example!")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: Shapley methods
    for idx, method in enumerate(SHAPLEY_METHODS):
        ax = axes[0, idx]
        doc_ids = shapley_query["document_ids"]
        attrs = shapley_query["attributions"][method]
        _plot_single_attribution(ax, doc_ids, attrs, method)
    
    # Row 2: Whitebox methods
    for idx, method in enumerate(WHITEBOX_METHODS):
        ax = axes[1, idx]
        doc_ids = whitebox_query["document_ids"]
        attrs = whitebox_query["attributions"][method]
        _plot_single_attribution(ax, doc_ids, attrs, method)
    
    # Add row labels
    fig.text(0.02, 0.72, "Shapley\nMethods", fontsize=12, fontweight="bold", 
             va="center", ha="center", rotation=90)
    fig.text(0.02, 0.28, "Whitebox\nMethods", fontsize=12, fontweight="bold", 
             va="center", ha="center", rotation=90)
    
    question = shapley_query.get("question", "")[:70]
    plt.suptitle(f"Attribution Scores for Query: \"{question}...\"", 
                fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved attribution example to {output_path}")


def _plot_single_attribution(ax, doc_ids: List[str], attributions: List[float], method: str):
    """Plot a single attribution subplot."""
    # Sort by absolute value
    sorted_indices = sorted(range(len(doc_ids)), key=lambda i: abs(attributions[i]), reverse=True)
    sorted_docs = [doc_ids[i] for i in sorted_indices]
    sorted_scores = [attributions[i] for i in sorted_indices]
    
    # Color by gold doc status
    colors = [COLORS["gold"] if doc in ["A", "B"] else COLORS["other"] for doc in sorted_docs]
    
    bars = ax.barh(range(len(sorted_docs)), sorted_scores, color=colors, 
                   alpha=0.85, edgecolor="black", linewidth=0.8)
    
    ax.set_yticks(range(len(sorted_docs)))
    ax.set_yticklabels(sorted_docs)
    ax.set_xlabel("Attribution Score", fontweight="bold")
    ax.set_ylabel("Document", fontweight="bold")
    ax.set_title(METHOD_NAMES.get(method, method), fontweight="bold", pad=8)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add value labels
    max_abs = max(abs(s) for s in sorted_scores) if sorted_scores else 1
    for i, (doc, score) in enumerate(zip(sorted_docs, sorted_scores)):
        offset = 0.02 * max_abs if score >= 0 else -0.02 * max_abs
        ha = "left" if score >= 0 else "right"
        ax.text(score + offset, i, f"{score:.2f}", va="center", ha=ha, fontsize=8)
    
    ax.invert_yaxis()


# ============================================================================
# PLOT 2: Accuracy vs Efficiency Tradeoff
# ============================================================================

def plot_accuracy_efficiency_tradeoff(
    shapley_results: List[Dict],
    whitebox_results: List[Dict],
    output_path: str
):
    """
    Create a scatter plot of accuracy vs efficiency for all methods.
    Uses real timing data from results.
    """
    ensure_dir(Path(output_path).parent)
    
    # Get metrics and timings
    shapley_metrics = get_aggregate_metrics(shapley_results)
    whitebox_metrics = get_aggregate_metrics(whitebox_results)
    all_metrics = {**shapley_metrics, **whitebox_metrics}
    
    shapley_timings = get_timing_data(shapley_results)
    whitebox_timings = get_timing_data(whitebox_results)
    all_timings = {**shapley_timings, **whitebox_timings}
    
    # Prepare data
    methods = []
    accuracies = []
    mean_times = []
    colors_list = []
    
    for method in SHAPLEY_METHODS + WHITEBOX_METHODS:
        if method in all_metrics and all_metrics[method]["top2_accuracy"] is not None:
            methods.append(method)
            accuracies.append(all_metrics[method]["top2_accuracy"])
            
            if method in all_timings and all_timings[method]:
                mean_times.append(np.mean(all_timings[method]))
            else:
                # Use placeholder if no timing data
                mean_times.append(1.0)
            
            colors_list.append(COLORS.get(method, "#333333"))
    
    if not methods:
        print("No data for accuracy vs efficiency plot!")
        return
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Computation time by method
    ax1 = axes[0]
    x_pos = np.arange(len(methods))
    bars = ax1.bar(x_pos, mean_times, color=colors_list, alpha=0.85, 
                   edgecolor="black", linewidth=1.2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=45, ha="right")
    ax1.set_ylabel("Mean Computation Time (s)", fontweight="bold")
    ax1.set_title("Computation Time by Method", fontweight="bold", pad=10)
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # Add value labels
    for bar, time in zip(bars, mean_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f"{time:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Right plot: Accuracy vs Efficiency scatter
    ax2 = axes[1]
    
    # Normalize efficiency (inverse of time, scaled 0-1)
    max_time = max(mean_times)
    efficiencies = [1 - (t / max_time) * 0.9 for t in mean_times]  # Scale so fastest is ~1, slowest ~0.1
    
    scatter = ax2.scatter(efficiencies, accuracies, s=300, c=colors_list, 
                         alpha=0.8, edgecolors="black", linewidth=2)
    
    # Add method labels
    for i, method in enumerate(methods):
        offset_x = 0.02
        offset_y = 0.02
        ax2.annotate(METHOD_NAMES.get(method, method).replace(" ", "\n"), 
                    (efficiencies[i], accuracies[i]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, ha="left", va="bottom")
    
    ax2.set_xlabel("Efficiency (Normalized)", fontweight="bold")
    ax2.set_ylabel("Top-2 Accuracy", fontweight="bold")
    ax2.set_title("Accuracy vs Efficiency Tradeoff", fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim([0, 1.15])
    ax2.set_ylim([0, 1.1])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    # Add quadrant labels
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved accuracy vs efficiency plot to {output_path}")


# ============================================================================
# PLOT 3: Method Comparison (2 rows)
# ============================================================================

def plot_method_comparison(
    shapley_results: List[Dict],
    whitebox_results: List[Dict],
    output_path: str
):
    """
    Create method comparison plot with 2 rows:
    - Row 1: Shapley methods
    - Row 2: Whitebox methods
    
    Each row has 3 subplots: Top-2 Accuracy, Mean Rank A, Mean Rank B
    """
    ensure_dir(Path(output_path).parent)
    
    shapley_metrics = get_aggregate_metrics(shapley_results)
    whitebox_metrics = get_aggregate_metrics(whitebox_results)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: Shapley methods
    _plot_method_row(axes[0], shapley_metrics, SHAPLEY_METHODS, "Shapley-based Methods")
    
    # Row 2: Whitebox methods
    _plot_method_row(axes[1], whitebox_metrics, WHITEBOX_METHODS, "Whitebox Methods")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved method comparison to {output_path}")


def _plot_method_row(axes, metrics: Dict, methods: List[str], row_title: str):
    """Plot a single row of method comparison (3 subplots)."""
    # Filter to methods that have data
    available_methods = [m for m in methods if m in metrics and metrics[m]["top2_accuracy"] is not None]
    
    if not available_methods:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    x_pos = np.arange(len(available_methods))
    colors_list = [COLORS.get(m, "#333333") for m in available_methods]
    labels = [METHOD_NAMES.get(m, m) for m in available_methods]
    
    # Plot 1: Top-2 Accuracy
    ax1 = axes[0]
    accs = [metrics[m]["top2_accuracy"] for m in available_methods]
    acc_stds = [metrics[m].get("top2_accuracy_std", 0) for m in available_methods]
    
    bars = ax1.bar(x_pos, accs, yerr=acc_stds, capsize=5, color=colors_list, 
                   alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Top-2 Accuracy", fontweight="bold")
    ax1.set_title(f"{row_title}: Top-2 Accuracy", fontweight="bold", pad=10)
    ax1.set_ylim([0, 1.15])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.03, f"{acc:.2f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Plot 2: Mean Rank A
    ax2 = axes[1]
    rank_a = [metrics[m]["mean_rank_A"] for m in available_methods]
    rank_a_stds = [metrics[m].get("mean_rank_A_std", 0) for m in available_methods]
    
    bars = ax2.bar(x_pos, rank_a, yerr=rank_a_stds, capsize=5, color=colors_list, 
                   alpha=0.85, edgecolor="black", linewidth=1.2)
    ax2.axhline(y=1.5, color="#2ca02c", linestyle="--", linewidth=2, 
               alpha=0.7, label="Ideal (1.5)", zorder=0)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Mean Rank of Document A", fontweight="bold")
    ax2.set_title(f"{row_title}: Mean Rank A", fontweight="bold", pad=10)
    max_rank = max(rank_a) if rank_a else 3
    ax2.set_ylim([0.8, max(3.5, max_rank + 0.5)])
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    for i, (bar, rank) in enumerate(zip(bars, rank_a)):
        ax2.text(bar.get_x() + bar.get_width()/2, rank + 0.1, f"{rank:.2f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Plot 3: Mean Rank B
    ax3 = axes[2]
    rank_b = [metrics[m]["mean_rank_B"] for m in available_methods]
    rank_b_stds = [metrics[m].get("mean_rank_B_std", 0) for m in available_methods]
    
    bars = ax3.bar(x_pos, rank_b, yerr=rank_b_stds, capsize=5, color=colors_list, 
                   alpha=0.85, edgecolor="black", linewidth=1.2)
    ax3.axhline(y=1.5, color="#2ca02c", linestyle="--", linewidth=2, 
               alpha=0.7, label="Ideal (1.5)", zorder=0)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=30, ha="right")
    ax3.set_ylabel("Mean Rank of Document B", fontweight="bold")
    ax3.set_title(f"{row_title}: Mean Rank B", fontweight="bold", pad=10)
    max_rank = max(rank_b) if rank_b else 3
    ax3.set_ylim([0.8, max(3.5, max_rank + 0.5)])
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    
    for i, (bar, rank) in enumerate(zip(bars, rank_b)):
        ax3.text(bar.get_x() + bar.get_width()/2, rank + 0.1, f"{rank:.2f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = Path("figures/final")
    ensure_dir(output_dir)
    
    # Define input files
    shapley_files = [
        "results/20_complementary_20251212_221510_full.json",
        "results/20_duplicate_20251212_221425_full.json",
        "results/20_synergy_20251212_222327_full.json",
    ]
    
    whitebox_files = [
        "results/whitebox_complementary_full.json",
        "results/whitebox_duplicate_full.json",
        "results/whitebox_synergy_full.json",
    ]
    
    # Load results
    print("Loading results...")
    shapley_results, whitebox_results = load_results(shapley_files, whitebox_files)
    print(f"Loaded {len(shapley_results)} Shapley result files")
    print(f"Loaded {len(whitebox_results)} whitebox result files")
    
    if not shapley_results and not whitebox_results:
        print("No results found!")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Attribution Example
    print("  1. Attribution example...")
    plot_attribution_example(
        shapley_results, 
        whitebox_results, 
        str(output_dir / "attribution_example.png")
    )
    
    # 2. Accuracy vs Efficiency Tradeoff
    print("  2. Accuracy vs efficiency tradeoff...")
    plot_accuracy_efficiency_tradeoff(
        shapley_results, 
        whitebox_results, 
        str(output_dir / "accuracy_efficiency_tradeoff.png")
    )
    
    # 3. Method Comparison
    print("  3. Method comparison...")
    plot_method_comparison(
        shapley_results, 
        whitebox_results, 
        str(output_dir / "method_comparison.png")
    )
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()

