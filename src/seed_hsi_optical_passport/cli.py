from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seed-hsi-passport",
        description="Run block-aware optical-passport validation for sorghum seed HSI tables.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory containing processed input tables.")
    parser.add_argument("--output", required=True, type=Path, help="Directory where output tables will be written.")
    parser.add_argument("--repeats", type=int, default=50, help="Number of repeated random stratified CV runs.")
    parser.add_argument("--permutations", type=int, default=200, help="Number of label permutations for random-CV null models.")
    parser.add_argument("--permutation-repeats", type=int, default=2, help="Repeated CV runs per label permutation.")
    parser.add_argument("--bootstrap", type=int, default=300, help="Bootstrap iterations for derivative-band stability selection.")
    parser.add_argument("--band-counts", default="1,3,5,10,20", help="Comma-separated minimal-band model sizes.")
    parser.add_argument("--seed", type=int, default=1729, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        repeats=args.repeats,
        permutations=args.permutations,
        permutation_repeats=args.permutation_repeats,
        bootstrap=args.bootstrap,
        band_counts=args.band_counts,
        seed=args.seed,
    )
    print("Analysis complete.")
    print(f"Summary: {outputs['summary']}")
    print(f"Workbook: {outputs['workbook']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
