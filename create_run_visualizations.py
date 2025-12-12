#!/usr/bin/env python3
"""
Deprecated wrapper. Use the unified CLI instead:
    python create_visualizations.py run --input <path> [--output-dir figures]
"""

import argparse

from visualizations.per_run import run_per_run


def main():
    parser = argparse.ArgumentParser(description="Wrapper for per-run visualizations.")
    parser.add_argument("--input", required=True, help="Path to *_full.json")
    parser.add_argument("--output-dir", default="figures", help="Directory to write figures")
    args = parser.parse_args()
    run_per_run(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

