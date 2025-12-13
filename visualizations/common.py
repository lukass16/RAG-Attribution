"""
Shared utilities for visualization scripts.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Global style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("colorblind")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    }
)

# Colorblind-friendly palette for consistent use
COLORS: Dict[str, str] = {
    "primary": "#0173B2",
    "secondary": "#DE8F05",
    "tertiary": "#029E73",
    "quaternary": "#CC78BC",
    "accent": "#56B4E9",
    "success": "#009E73",
    "warning": "#F0E442",
    "error": "#D55E00",
    "gold": "#E69F00",
    "red": "#D55E00",
    "blue": "#0173B2",
}


def format_method_name(method_name: str) -> str:
    """Format method name for display (e.g., 'leave_one_out' -> 'Leave-One-Out')."""
    name_map = {
        "leave_one_out": "Leave-One-Out",
        "permutation_shapley": "Permutation Shapley",
        "monte_carlo_shapley": "Monte Carlo Shapley",
        "kernel_shap": "Kernel SHAP",
    }
    return name_map.get(method_name, method_name.replace("_", " ").title())


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_results(results_dir: str = "results", files: Optional[List[str]] = None) -> List[Dict]:
    """Load *_full.json result files."""
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
    for csv_file in metrics_path.glob("combined_metrics_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            frames.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

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

    return pd.concat(frames, ignore_index=True, sort=False)

