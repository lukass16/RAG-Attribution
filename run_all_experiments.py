#!/usr/bin/env python3
"""
Run multiple attribution experiments in sequence and write a combined metrics CSV.

Default datasets:
- data/20_complementary.csv
- data/20_synergy.csv
- data/20_duplicate.csv
"""

import argparse
import os
from datetime import datetime
from typing import List

import pandas as pd

from run_attribution_experiments import run_attribution_experiment, save_results


def run_batch(datasets: List[str], output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    combined_rows = []

    for ds in datasets:
        print("\n" + "=" * 80)
        print(f"Running dataset: {ds}")
        print("=" * 80)

        results = run_attribution_experiment(dataset_path=ds)
        save_results(results, output_dir=output_dir)

        timestamp = results.get("timestamp", datetime.utcnow().isoformat())
        for method, metrics in results["aggregate_metrics"].items():
            combined_rows.append(
                {
                    "dataset": ds,
                    "method": method,
                    "top2_accuracy": metrics.get("top2_accuracy"),
                    "mean_rank_A": metrics.get("mean_rank_A"),
                    "mean_rank_B": metrics.get("mean_rank_B"),
                    "n_queries": metrics.get("n_queries"),
                    "timestamp": timestamp,
                }
            )

    combined_df = pd.DataFrame(combined_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(output_dir, f"combined_metrics_{ts}.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined metrics to: {combined_path}")
    return combined_path


def main():
    parser = argparse.ArgumentParser(description="Run multiple attribution experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "data/20_complementary.csv",
            "data/20_synergy.csv",
            "data/20_duplicate.csv",
        ],
        help="List of dataset paths to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to store result files.",
    )
    args = parser.parse_args()

    run_batch(args.datasets, args.output_dir)


if __name__ == "__main__":
    main()

