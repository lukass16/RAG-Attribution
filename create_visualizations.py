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
