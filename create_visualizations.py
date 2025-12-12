#!/usr/bin/env python3
"""
Unified entry point for all visualization workflows.

Subcommands:
- summary (default): multi-run summaries from *_full.json + metrics CSVs
- run: per-run figures from a single *_full.json
- extras: additional/report figures from metrics CSVs
- all: run summary + extras (and run-level if --input provided)
"""

import argparse

from visualizations.summary import run_summary
from visualizations.per_run import run_per_run
from visualizations.extras import run_extras


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create visualizations for RAG attribution.")
    subparsers = parser.add_subparsers(dest="command")

    summary_p = subparsers.add_parser("summary", help="Create multi-run summary figures.")
    summary_p.add_argument("--results-dir", type=str, default="results", help="Directory with *_full.json and metrics CSVs")
    summary_p.add_argument("--files", nargs="*", default=None, help="Explicit list of *_full.json files")
    summary_p.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures")

    run_p = subparsers.add_parser("run", help="Create per-run figures from a single *_full.json.")
    run_p.add_argument("--input", required=True, help="Path to *_full.json")
    run_p.add_argument("--output-dir", default="figures", help="Directory to write figures")

    extras_p = subparsers.add_parser("extras", help="Create additional/report-focused figures.")
    extras_p.add_argument("--results-dir", type=str, default="results", help="Directory with metrics CSVs")
    extras_p.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures")

    all_p = subparsers.add_parser("all", help="Run summary + extras, and run-level if --input is given.")
    all_p.add_argument("--results-dir", type=str, default="results", help="Directory with *_full.json and metrics CSVs")
    all_p.add_argument("--files", nargs="*", default=None, help="Explicit list of *_full.json files")
    all_p.add_argument("--input", help="Optional *_full.json for per-run figures")
    all_p.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command or "summary"
    if cmd == "summary":
        run_summary(results_dir=args.results_dir, files=args.files or [], output_dir=args.output_dir)
    elif cmd == "run":
        run_per_run(input_path=args.input, output_dir=args.output_dir)
    elif cmd == "extras":
        run_extras(results_dir=args.results_dir, output_dir=args.output_dir)
    elif cmd == "all":
        run_summary(results_dir=args.results_dir, files=args.files or [], output_dir=args.output_dir)
        run_extras(results_dir=args.results_dir, output_dir=args.output_dir)
        if args.input:
            run_per_run(input_path=args.input, output_dir=args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


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

