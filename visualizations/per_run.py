"""
Per-run visualizations (single *_full.json).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualizations.common import format_method_name, ensure_dir

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def load_full_run(path: Path) -> Dict:
    import json

    with path.open() as f:
        return json.load(f)


def as_list_attributions(attrs) -> List[float]:
    if isinstance(attrs, dict):
        n = max(attrs.keys()) + 1 if attrs else 0
        return [attrs.get(i, 0.0) for i in range(n)]
    return list(attrs)


def precision_recall_at_k(attributions: List[float], doc_ids: List[str], gold: List[str], k: int) -> Tuple[float, float]:
    pairs = list(zip(doc_ids, attributions))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    topk = [d for d, _ in pairs[:k]]
    gold_set = set(gold)
    hits = sum(1 for d in topk if d in gold_set)
    prec = hits / max(len(topk), 1)
    rec = hits / max(len(gold_set), 1)
    return prec, rec


def build_frames(run: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    methods = run["methods"]
    rows = []
    pr_rows = []
    timing_rows = []
    for q in run["results"]:
        doc_ids = q["document_ids"]
        gold_docs = q.get("gold_docs", ["A", "B"])
        for m in methods:
            attrs = as_list_attributions(q["attributions"][m])
            rank_A = q.get(f"{m}_rank_A")
            rank_B = q.get(f"{m}_rank_B")
            top2 = q.get(f"{m}_top2_accuracy")
            timing = q.get("timings_seconds", {}).get(m)
            rows.append({"query_idx": q["query_idx"], "method": m, "rank_A": rank_A, "rank_B": rank_B, "top2": top2})
            if timing is not None:
                timing_rows.append({"method": m, "timing_seconds": timing})
            for k in (3, 5):
                p, r = precision_recall_at_k(attrs, doc_ids, gold_docs, k)
                pr_rows.append({"method": m, "k": k, "precision": p, "recall": r})
    df = pd.DataFrame(rows)
    df_pr = pd.DataFrame(pr_rows)
    df_timing = pd.DataFrame(timing_rows)
    return df, df_pr, df_timing, methods


def plot_mean_ranks(df: pd.DataFrame, output_path: Path, method_order: List[str]):
    agg = (
        df.groupby("method")[["rank_A", "rank_B"]]
        .mean()
        .reset_index()
        .melt(id_vars="method", value_vars=["rank_A", "rank_B"], var_name="doc", value_name="mean_rank")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=agg, x="method", y="mean_rank", hue="doc", palette="Set2", order=method_order)
    plt.title("Mean rank of gold docs (lower is better)", fontweight="bold")
    plt.xlabel("Method")
    plt.ylabel("Mean rank")
    plt.ylim(1, max(agg["mean_rank"].max(), 2.5))
    plt.legend(title="Doc")
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width() / 2, height + 0.05, f"{height:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rank_box(df: pd.DataFrame, output_path: Path, method_order: List[str]):
    melted = df.melt(
        id_vars=["method", "query_idx", "top2"],
        value_vars=["rank_A", "rank_B"],
        var_name="doc",
        value_name="rank",
    )
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=melted, x="method", y="rank", hue="doc", palette="Set2", order=method_order)
    plt.title("Rank distribution of gold docs", fontweight="bold")
    plt.xlabel("Method")
    plt.ylabel("Rank (lower is better)")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_timings(df_timing: pd.DataFrame, output_path: Path, method_order: List[str]):
    if df_timing.empty:
        return
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(
        data=df_timing,
        x="method",
        y="timing_seconds",
        hue="method",
        order=method_order,
        palette="pastel",
        legend=False,
    )
    plt.title("Per-query runtime by method (seconds)", fontweight="bold")
    plt.xlabel("Method")
    plt.ylabel("Seconds")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_precision_recall(df_pr: pd.DataFrame, output_path: Path, method_order: List[str]):
    agg = (
        df_pr.groupby(["method", "k"])
        .agg({"precision": "mean", "recall": "mean"})
        .reset_index()
        .melt(id_vars=["method", "k"], value_vars=["precision", "recall"], var_name="metric", value_name="value")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=agg, x="method", y="value", hue="metric", palette="muted", dodge=True, order=method_order)
    plt.title("Precision / Recall at K", fontweight="bold")
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width() / 2, height + 0.02, f"{height:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_per_run(input_path: str, output_dir: str):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    run = load_full_run(path)
    df, df_pr, df_timing, method_order = build_frames(run)

    prefix = path.stem.replace("_full", "")
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    plot_mean_ranks(df, out_dir / f"{prefix}_mean_ranks.png", method_order)
    plot_rank_box(df, out_dir / f"{prefix}_rank_box.png", method_order)
    plot_timings(df_timing, out_dir / f"{prefix}_timings.png", method_order)
    plot_precision_recall(df_pr, out_dir / f"{prefix}_precision_recall.png", method_order)
