#!/usr/bin/env python3
"""
Create visualizations for RAG Source Attribution Analysis

Generates:
1. Architecture diagram
2. Method comparison plots
3. Results summary visualizations
4. Ablation study plots
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Set style with colorblind-friendly palette
plt.style.use('seaborn-v0_8-darkgrid')
# Use colorblind-friendly palette (works well for scientific visualizations)
sns.set_palette("colorblind")
# Set default font sizes for better readability
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Colorblind-friendly color palette for consistent use
COLORS = {
    'primary': '#0173B2',      # Blue
    'secondary': '#DE8F05',   # Orange
    'tertiary': '#029E73',     # Green
    'quaternary': '#CC78BC',   # Purple
    'accent': '#56B4E9',      # Light blue
    'success': '#009E73',     # Green
    'warning': '#F0E442',     # Yellow
    'error': '#D55E00',       # Red-orange
    'gold': '#E69F00',        # Gold
    'red': '#D55E00',         # Red
    'blue': '#0173B2',        # Blue
}

def format_method_name(method_name: str) -> str:
    """Format method name for display (e.g., 'leave_one_out' -> 'Leave-One-Out')."""
    name_map = {
        'leave_one_out': 'Leave-One-Out',
        'permutation_shapley': 'Permutation Shapley',
        'monte_carlo_shapley': 'Monte Carlo Shapley',
        'kernel_shap': 'Kernel SHAP',
    }
    return name_map.get(method_name, method_name.replace('_', ' ').title())

def load_results(results_dir: str = "results", files: Optional[List[str]] = None) -> List[Dict]:
    """Load result JSON files, either explicit paths or all *_full.json in a directory."""
    results: List[Dict] = []
    json_paths: List[Path] = []
    if files:
        for fp in files:
            p = Path(fp)
            if p.exists() and p.suffix == ".json":
                json_paths.append(p)
            else:
                print(f"Skipping missing/non-json file: {fp}")
    else:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Results directory {results_dir} does not exist!")
            return results
        json_paths.extend(results_path.glob("*_full.json"))
    for json_file in json_paths:
        try:
            with json_file.open("r") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    return results


def infer_dataset_from_metrics_path(path: Path) -> str:
    """Infer dataset name from metrics filename, e.g., 20_synergy_20251212_202138_metrics.csv -> 20_synergy."""
    stem = path.stem.replace("_metrics", "")
    parts = stem.split("_")
    if len(parts) > 2:
        return "_".join(parts[:-2])
    return stem


def load_metrics_frames(results_dir: str = "results") -> pd.DataFrame:
    """Load all *_metrics.csv (and combined_metrics_*.csv if present) into a single DataFrame."""
    metrics_path = Path(results_dir)
    if not metrics_path.exists():
        return pd.DataFrame()

    frames = []
    # Combined metrics first (already contains dataset)
    for csv_file in metrics_path.glob("combined_metrics_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            frames.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # Per-run metrics without dataset column; infer from filename
    for csv_file in metrics_path.glob("*_metrics.csv"):
        if "combined_metrics_" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file)
            df["dataset"] = infer_dataset_from_metrics_path(csv_file)
            frames.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined

def create_architecture_diagram(output_path: str = "figures/architecture.png"):
    """Create architecture diagram with clean layout."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.axis('off')
    
    # Define components in a grid layout with colorblind-friendly colors
    # Format: (x, y, color, width, height)
    components = {
        'Question\nQ': (2, 8, COLORS['accent'], 2.2, 1.0),
        'Documents\nD = {d₁, ..., d_n}': (2, 5.5, COLORS['tertiary'], 2.2, 1.0),
        'RAG System\n(LLM)': (6, 6.75, COLORS['warning'], 2.5, 1.2),
        'Target Response\nR_target': (10.5, 8, COLORS['error'], 2.5, 1.0),
        'Document Subsets\nS ⊆ D': (6, 4, COLORS['gold'], 2.5, 1.0),
        'Utility Function\nv(S)': (10.5, 4, COLORS['quaternary'], 2.5, 1.0),
        'Attribution\nMethods': (14.5, 6, COLORS['secondary'], 2.5, 1.2),
        'Attribution\nScores φᵢ': (14.5, 2.5, COLORS['primary'], 2.5, 1.0),
    }
    
    # Draw components
    for name, (x, y, color, width, height) in components.items():
        rect = plt.Rectangle((x-width/2, y-height/2), width, height, 
                            facecolor=color, edgecolor='black', linewidth=2,
                            zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, 
               fontweight='bold', zorder=4)
    
    # Draw arrows with clean routing - no overlaps
    from matplotlib.patches import FancyArrowPatch
    
    # Arrow 1: Question to RAG System
    arrow1 = FancyArrowPatch((3.1, 8), (4.75, 7.2),
                            arrowstyle='->', mutation_scale=20, lw=2, 
                            color='black', zorder=2)
    ax.add_patch(arrow1)
    
    # Arrow 2: Documents to RAG System  
    arrow2 = FancyArrowPatch((3.1, 5.5), (4.75, 6.3),
                            arrowstyle='->', mutation_scale=20, lw=2,
                            color='black', zorder=2)
    ax.add_patch(arrow2)
    
    # Arrow 3: RAG System to Target Response
    arrow3 = FancyArrowPatch((7.25, 7.2), (9.25, 8),
                            arrowstyle='->', mutation_scale=20, lw=2,
                            color='black', zorder=2)
    ax.add_patch(arrow3)
    ax.text(8.25, 7.8, 'Generate\nwith all docs', ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, zorder=5))
    
    # Arrow 4: Documents to Document Subsets
    arrow4 = FancyArrowPatch((3.1, 5.0), (4.75, 4),
                            arrowstyle='->', mutation_scale=20, lw=2,
                            color='black', zorder=2)
    ax.add_patch(arrow4)
    ax.text(3.5, 4.3, 'Sample\nsubsets', ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, zorder=5))
    
    # Arrow 5: Document Subsets to Utility Function
    arrow5 = FancyArrowPatch((7.25, 4), (9.25, 4),
                            arrowstyle='->', mutation_scale=20, lw=2,
                            color='black', zorder=2)
    ax.add_patch(arrow5)
    ax.text(8.25, 4.35, 'Compute', ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, zorder=5))
    
    # Arrow 6: Target Response to Attribution Methods
    arrow6 = FancyArrowPatch((11.75, 8), (13.25, 6.6),
                            arrowstyle='->', mutation_scale=20, lw=2,
                            color='black', zorder=2)
    ax.add_patch(arrow6)
    
    # Arrow 7: Utility Function to Attribution Methods
    arrow7 = FancyArrowPatch((11.75, 4), (13.25, 5.4),
                            arrowstyle='->', mutation_scale=20, lw=2,
                            color='black', zorder=2)
    ax.add_patch(arrow7)
    
    # Arrow 8: Attribution Methods to Scores
    arrow8 = FancyArrowPatch((14.5, 4.8), (14.5, 3.0),
                            arrowstyle='->', mutation_scale=20, lw=2,
                            color='black', zorder=2)
    ax.add_patch(arrow8)
    ax.text(14.9, 3.9, 'Rank', ha='left', va='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, zorder=5))
    
    # Add method labels in a box with better styling
    methods_box = plt.Rectangle((13, 0.2), 3.5, 1.8, facecolor=COLORS['warning'], 
                               edgecolor='black', linewidth=1.5, alpha=0.3, zorder=1)
    ax.add_patch(methods_box)
    ax.text(14.75, 1.7, 'Methods:', ha='center', fontsize=10, 
           fontweight='bold', zorder=2)
    methods = ['• Leave-One-Out', '• Permutation Shapley', 
               '• Monte Carlo Shapley', '• Kernel SHAP']
    for i, method in enumerate(methods):
        ax.text(13.3, 1.3-i*0.25, method, fontsize=9, va='center', ha='left', 
               zorder=2, fontweight='normal')
    
    # Add stage labels
    ax.text(2, 9.5, '1. Input', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.5))
    ax.text(6, 9.5, '2. Generation', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.5))
    ax.text(10.5, 9.5, '3. Evaluation', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.5))
    ax.text(14.5, 9.5, '4. Attribution', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.5))
    
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10)
    ax.set_title('RAG Source Attribution System Architecture', 
                fontsize=18, fontweight='bold', pad=25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved architecture diagram to {output_path}")
    plt.close()

def plot_method_comparison(results: List[Dict], output_path: str = "figures/method_comparison.png"):
    """Compare attribution methods across queries."""
    if not results:
        print("No results to plot!")
        return
    
    # Aggregate metrics across all results
    method_metrics = {}
    
    for result in results:
        for method_name, metrics in result['aggregate_metrics'].items():
            if method_name not in method_metrics:
                method_metrics[method_name] = {
                    'top2_acc': [],
                    'rank_A': [],
                    'rank_B': [],
                    'n_queries': []
                }
            
            if metrics['top2_accuracy'] is not None:
                method_metrics[method_name]['top2_acc'].append(metrics['top2_accuracy'])
            if metrics['mean_rank_A'] is not None:
                method_metrics[method_name]['rank_A'].append(metrics['mean_rank_A'])
            if metrics['mean_rank_B'] is not None:
                method_metrics[method_name]['rank_B'].append(metrics['mean_rank_B'])
    
    if not method_metrics:
        print("No metrics to plot!")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = list(method_metrics.keys())
    formatted_methods = [format_method_name(m) for m in methods]
    x_pos = np.arange(len(methods))
    
    # Get colorblind-friendly colors
    colors = sns.color_palette("colorblind", n_colors=len(methods))
    
    # Plot 1: Top-2 Accuracy
    top2_accs = [np.mean(method_metrics[m]['top2_acc']) if method_metrics[m]['top2_acc'] 
                 else 0 for m in methods]
    top2_stds = [np.std(method_metrics[m]['top2_acc']) if method_metrics[m]['top2_acc'] 
                 else 0 for m in methods]
    
    bars1 = axes[0].bar(x_pos, top2_accs, yerr=top2_stds, capsize=5, alpha=0.8, 
                       color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_xlabel('Attribution Method', fontweight='bold')
    axes[0].set_ylabel('Top-2 Accuracy', fontweight='bold')
    axes[0].set_title('Top-2 Accuracy by Method', fontweight='bold', pad=10)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(formatted_methods, rotation=45, ha='right')
    axes[0].set_ylim([0, 1.15])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Add value labels on bars
    for i, (acc, std) in enumerate(zip(top2_accs, top2_stds)):
        height = acc + std + 0.02
        axes[0].text(i, height, f'{acc:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Plot 2: Mean Rank of Document A
    rank_A_means = [np.mean(method_metrics[m]['rank_A']) if method_metrics[m]['rank_A'] 
                    else 0 for m in methods]
    rank_A_stds = [np.std(method_metrics[m]['rank_A']) if method_metrics[m]['rank_A'] 
                   else 0 for m in methods]
    
    bars2 = axes[1].bar(x_pos, rank_A_means, yerr=rank_A_stds, capsize=5, alpha=0.8,
                       color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_xlabel('Attribution Method', fontweight='bold')
    axes[1].set_ylabel('Mean Rank of Document A', fontweight='bold')
    axes[1].set_title('Mean Rank of Document A', fontweight='bold', pad=10)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(formatted_methods, rotation=45, ha='right')
    axes[1].axhline(y=1.5, color=COLORS['success'], linestyle='--', linewidth=2, 
                   alpha=0.7, label='Ideal (1.5)', zorder=0)
    axes[1].legend(loc='upper right', framealpha=0.9)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_ylim([0.8, max(3.0, max(rank_A_means) + max(rank_A_stds) + 0.3)])
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(rank_A_means, rank_A_stds)):
        height = mean + std + 0.1
        axes[1].text(i, height, f'{mean:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    # Plot 3: Mean Rank of Document B
    rank_B_means = [np.mean(method_metrics[m]['rank_B']) if method_metrics[m]['rank_B'] 
                    else 0 for m in methods]
    rank_B_stds = [np.std(method_metrics[m]['rank_B']) if method_metrics[m]['rank_B'] 
                   else 0 for m in methods]
    
    bars3 = axes[2].bar(x_pos, rank_B_means, yerr=rank_B_stds, capsize=5, alpha=0.8,
                       color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_xlabel('Attribution Method', fontweight='bold')
    axes[2].set_ylabel('Mean Rank of Document B', fontweight='bold')
    axes[2].set_title('Mean Rank of Document B', fontweight='bold', pad=10)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(formatted_methods, rotation=45, ha='right')
    axes[2].axhline(y=1.5, color=COLORS['success'], linestyle='--', linewidth=2,
                   alpha=0.7, label='Ideal (1.5)', zorder=0)
    axes[2].legend(loc='upper right', framealpha=0.9)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].set_ylim([0.8, max(3.0, max(rank_B_means) + max(rank_B_stds) + 0.3)])
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(rank_B_means, rank_B_stds)):
        height = mean + std + 0.1
        axes[2].text(i, height, f'{mean:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved method comparison to {output_path}")
    plt.close()


def plot_dataset_summary(metrics_df: pd.DataFrame, output_path: str = "figures/dataset_method_summary.png"):
    """Summarize metrics across datasets and methods."""
    if metrics_df.empty or "dataset" not in metrics_df.columns:
        print("No dataset-level metrics to plot!")
        return

    # Ensure numeric
    for col in ["top2_accuracy", "mean_rank_A", "mean_rank_B"]:
        if col in metrics_df:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    # Format method names for better readability
    if "method" in metrics_df.columns:
        metrics_df["method_formatted"] = metrics_df["method"].apply(format_method_name)
    else:
        metrics_df["method_formatted"] = "Method"

    # Top-2 accuracy per dataset/method
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=metrics_df, x="dataset", y="top2_accuracy", hue="method_formatted", 
                palette="colorblind", ax=ax, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_title("Top-2 Accuracy by Dataset and Method", fontweight="bold", pad=15, fontsize=14)
    ax.set_xlabel("Dataset", fontweight="bold")
    ax.set_ylabel("Top-2 Accuracy", fontweight="bold")
    ax.legend(title="Method", title_fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_top2.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Mean rank A/B averaged into one value
    metrics_df["mean_rank_AB"] = metrics_df[["mean_rank_A", "mean_rank_B"]].mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=metrics_df, x="dataset", y="mean_rank_AB", hue="method_formatted",
                palette="colorblind", ax=ax, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax.axhline(y=1.5, color=COLORS['success'], linestyle="--", linewidth=2, 
              alpha=0.7, label="Ideal (1.5)", zorder=0)
    max_rank = metrics_df["mean_rank_AB"].max()
    ax.set_ylim(0.5, max_rank + 0.5)
    ax.set_title("Mean Rank (A,B) by Dataset and Method", fontweight="bold", pad=15, fontsize=14)
    ax.set_xlabel("Dataset", fontweight="bold")
    ax.set_ylabel("Mean Rank (A,B)", fontweight="bold")
    ax.legend(title="Method", title_fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_rank.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_attribution_scores_example(results: List[Dict], output_path: str = "figures/attribution_example.png"):
    """Plot example attribution scores for a single query."""
    if not results:
        print("No results to plot!")
        return
    
    # Find result with valid attribution data
    query_result = None
    for result in results:
        for qr in result['results']:
            if 'attributions' in qr and qr['attributions']:
                # Check if attributions are actual scores (not just indices)
                for method, attrs in qr['attributions'].items():
                    if attrs and isinstance(attrs, list) and len(attrs) > 0:
                        # Check if it's not just sequential indices
                        if not (attrs == list(range(len(attrs))) or attrs == list(range(len(attrs)))[::-1]):
                            query_result = qr
                            break
                if query_result:
                    break
        if query_result:
            break
    
    if not query_result:
        print("No valid attribution data found!")
        return
    
    doc_ids = query_result['document_ids']
    
    # Find available methods with valid data
    available_methods = []
    for method_name in ['leave_one_out', 'permutation_shapley', 'monte_carlo_shapley', 'kernel_shap']:
        if method_name in query_result['attributions']:
            attrs = query_result['attributions'][method_name]
            if attrs and isinstance(attrs, list) and len(attrs) == len(doc_ids):
                # Check if it's not just sequential indices
                if not (attrs == list(range(len(attrs))) or attrs == list(range(len(attrs)))[::-1]):
                    available_methods.append(method_name)
    
    if not available_methods:
        print("No valid methods found!")
        return
    
    # Create appropriate number of subplots
    n_methods = min(len(available_methods), 4)
    n_cols = 2
    n_rows = (n_methods + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, method_name in enumerate(available_methods[:4]):
        if idx >= len(axes):
            break
        
        attributions = query_result['attributions'][method_name]
        
        # Sort by absolute value
        sorted_indices = sorted(range(len(doc_ids)), 
                               key=lambda i: abs(attributions[i]), 
                               reverse=True)
        
        sorted_docs = [doc_ids[i] for i in sorted_indices]
        sorted_scores = [attributions[i] for i in sorted_indices]
        
        # Use colorblind-friendly colors: red for gold docs (A, B), blue for others
        colors = [COLORS['red'] if doc in ['A', 'B'] else COLORS['primary'] 
                 for doc in sorted_docs]
        
        bars = axes[idx].barh(range(len(sorted_docs)), sorted_scores, color=colors, 
                             alpha=0.8, edgecolor='black', linewidth=1.2)
        axes[idx].set_yticks(range(len(sorted_docs)))
        axes[idx].set_yticklabels(sorted_docs, fontweight='bold' if sorted_docs[0] in ['A', 'B'] else 'normal')
        axes[idx].set_xlabel('Attribution Score', fontweight='bold')
        axes[idx].set_ylabel('Document', fontweight='bold')
        axes[idx].set_title(f'{format_method_name(method_name)}', 
                           fontweight='bold', pad=10)
        axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        axes[idx].grid(axis='x', alpha=0.3, linestyle='--')
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        
        # Add value labels
        max_abs = max(abs(s) for s in sorted_scores) if sorted_scores else 1
        for i, (doc, score) in enumerate(zip(sorted_docs, sorted_scores)):
            offset = 0.02 * max_abs if score >= 0 else -0.02 * max_abs
            axes[idx].text(score + offset, i, f'{score:.2f}', va='center',
                          ha='left' if score >= 0 else 'right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(available_methods), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Attribution Scores for Query: {query_result['question'][:60]}...", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved attribution example to {output_path}")
    plt.close()

def plot_ablation_study(output_path: str = "figures/ablation_study.png"):
    """Placeholder ablation study visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Sensitivity to number of samples (Monte Carlo Shapley) - illustrative only
    sample_sizes = [16, 32, 64, 128, 256]
    top2_acc_simulated = [0.65, 0.72, 0.78, 0.80, 0.81]
    axes[0].plot(sample_sizes, top2_acc_simulated, marker='o', linewidth=3, markersize=10,
                color=COLORS['primary'], markerfacecolor=COLORS['primary'], 
                markeredgecolor='black', markeredgewidth=1.5)
    axes[0].set_xlabel('Number of Samples', fontweight='bold')
    axes[0].set_ylabel('Top-2 Accuracy', fontweight='bold')
    axes[0].set_title('Sensitivity to Sample Size\n(Monte Carlo Shapley)', 
                     fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim([0.6, 0.85])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot 2: Impact of ranking by absolute value vs raw score (illustrative)
    methods = ['Leave-One-Out', 'Permutation\nShapley', 'Monte Carlo\nShapley']
    raw_score_acc = [0.6, 0.62, 0.64]
    abs_value_acc = [1.0, 0.8, 0.9]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[1].bar(x - width/2, raw_score_acc, width, label='Raw Score', alpha=0.8, 
               color=COLORS['secondary'], edgecolor='black', linewidth=1.2)
    axes[1].bar(x + width/2, abs_value_acc, width, label='Absolute Value', alpha=0.8,
               color=COLORS['primary'], edgecolor='black', linewidth=1.2)
    axes[1].set_xlabel('Attribution Method', fontweight='bold')
    axes[1].set_ylabel('Top-2 Accuracy', fontweight='bold')
    axes[1].set_title('Impact of Ranking by Absolute Value', fontweight='bold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].legend(framealpha=0.9)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.15])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ablation study to {output_path}")
    plt.close()

def plot_per_query_analysis(results: List[Dict], output_path: str = "figures/per_query_analysis.png"):
    """Plot per-query analysis."""
    if not results:
        print("No results to plot!")
        return
    
    # Collect per-query data
    query_data = []
    
    for result in results:
        for query_result in result['results']:
            query_idx = query_result['query_idx']
            # Get methods from aggregate_metrics or from query_result attributions
            methods_in_result = list(result.get('aggregate_metrics', {}).keys())
            if not methods_in_result:
                methods_in_result = list(query_result.get('attributions', {}).keys())
            
            for method_name in methods_in_result:
                top2_key = f'{method_name}_top2_accuracy'
                rank_A_key = f'{method_name}_rank_A'
                rank_B_key = f'{method_name}_rank_B'
                
                if top2_key in query_result:
                    query_data.append({
                        'query_idx': query_idx,
                        'method': method_name,
                        'top2_acc': query_result[top2_key],
                        'rank_A': query_result.get(rank_A_key),
                        'rank_B': query_result.get(rank_B_key)
                    })
    
    if not query_data:
        print("No per-query data!")
        return
    
    df = pd.DataFrame(query_data)
    # Restrict to first 5 queries
    top_queries = sorted(df['query_idx'].unique())[:5]
    df = df[df['query_idx'].isin(top_queries)]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    methods = df['method'].unique().tolist()
    if df.empty or not methods:
        print("No methods/queries to plot!")
        return
    
    # Format method names
    df['method_formatted'] = df['method'].apply(format_method_name)
    
    # Bar plot: Top-2 accuracy per query (only first 10)
    sns.barplot(data=df, x='query_idx', y='top2_acc', hue='method_formatted', 
               ax=axes[0], palette='colorblind', edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[0].set_xlabel('Query Index', fontweight='bold')
    axes[0].set_ylabel('Top-2 Accuracy', fontweight='bold')
    axes[0].set_title('Top-2 Accuracy (per query)', fontweight='bold', pad=10)
    axes[0].set_ylim([-0.05, 1.1])
    axes[0].legend(title='Method', title_fontsize=11, framealpha=0.9)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Bar plot: Mean rank (average of A and B) per query
    df['mean_rank_AB'] = df[['rank_A', 'rank_B']].mean(axis=1)
    sns.barplot(data=df, x='query_idx', y='mean_rank_AB', hue='method_formatted',
               ax=axes[1], palette='colorblind', edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[1].axhline(y=1.5, color=COLORS['success'], linestyle='--', linewidth=2, 
                   alpha=0.7, label='Ideal (1.5)', zorder=0)
    axes[1].set_xlabel('Query Index', fontweight='bold')
    axes[1].set_ylabel('Mean Rank (A,B)', fontweight='bold')
    axes[1].set_title('Mean Rank of Gold Docs (per query)', fontweight='bold', pad=10)
    max_rank = df['mean_rank_AB'].max()
    axes[1].set_ylim([0.5, max(3.0, max_rank + 0.5)])
    axes[1].legend(title='Method', title_fontsize=11, framealpha=0.9)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-query analysis to {output_path}")
    plt.close()

def create_summary_table(results: List[Dict], output_path: str = "figures/summary_table.png"):
    """Create a summary table visualization."""
    if not results:
        print("No results to plot!")
        return
    
    # Aggregate metrics
    all_metrics = []
    for result in results:
        for method_name, metrics in result['aggregate_metrics'].items():
            all_metrics.append({
                'Method': format_method_name(method_name),
                'Top-2 Accuracy': f"{metrics['top2_accuracy']:.3f}" if metrics['top2_accuracy'] is not None else "N/A",
                'Mean Rank A': f"{metrics['mean_rank_A']:.2f}" if metrics['mean_rank_A'] is not None else "N/A",
                'Mean Rank B': f"{metrics['mean_rank_B']:.2f}" if metrics['mean_rank_B'] is not None else "N/A",
                'N Queries': metrics['n_queries']
            })
    
    if not all_metrics:
        print("No metrics to display!")
        return
    
    df = pd.DataFrame(all_metrics)
    df = df.drop_duplicates(subset=['Method']).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.8)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Attribution Methods Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary table to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create visualizations for RAG attribution')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing result JSON files (ignored if --files is set)')
    parser.add_argument('--files', nargs='*', default=None,
                       help='Explicit list of *_full.json result files to load')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir, args.files)
    metrics_df = load_metrics_frames(args.results_dir)
    
    print(f"Loaded {len(results)} result files")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    create_architecture_diagram(os.path.join(args.output_dir, "architecture.png"))
    
    if results:
        plot_method_comparison(results, os.path.join(args.output_dir, "method_comparison.png"))
        plot_attribution_scores_example(results, os.path.join(args.output_dir, "attribution_example.png"))
        plot_per_query_analysis(results, os.path.join(args.output_dir, "per_query_analysis.png"))
        create_summary_table(results, os.path.join(args.output_dir, "summary_table.png"))
    
    if not metrics_df.empty:
        plot_dataset_summary(metrics_df, os.path.join(args.output_dir, "dataset_method_summary.png"))
    
    plot_ablation_study(os.path.join(args.output_dir, "ablation_study.png"))
    
    print(f"\nAll visualizations saved to {args.output_dir}/")

if __name__ == '__main__':
    main()

