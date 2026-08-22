"""
CLI entry point for SnapTidy V1.

Usage:
    python -m snap_tidy process /path/to/photos --window week --output result.json

Provides argument parsing, logging setup, and calls pipeline.run_pipeline().
All testing must use synthetic images — never real photo libraries.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        prog="snap_tidy",
        description="AI-assisted photo cleanup tool — backend pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── process command ────────────────────────────────────────────────
    proc = subparsers.add_parser("process", help="Process a directory of photos")
    proc.add_argument("source", type=str, help="Directory containing photos to process")
    proc.add_argument(
        "--window", "-w",
        choices=["day", "week", "month", "year"],
        default="week",
        help="Date grouping granularity (default: week)",
    )
    proc.add_argument(
        "--min-cluster-size", "-m",
        type=int, default=5,
        help="Minimum cluster size for HDBSCAN (default: 5)",
    )
    proc.add_argument(
        "--quality-threshold", "-q",
        type=float, default=60.0,
        help="Quality score threshold for keep vs archive (default: 60)",
    )
    proc.add_argument(
        "--top-k", "-k",
        type=int, default=5,
        help="Number of top photos to preview per group (default: 5)",
    )
    proc.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output JSON file path (default: stdout)",
    )
    proc.add_argument(
        "--device", "-d",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Compute device (default: auto-detect)",
    )
    proc.add_argument(
        "--batch-size", "-b",
        type=int, default=32,
        help="CLIP batch size (default: 32)",
    )
    proc.add_argument(
        "--dhash-threshold",
        type=int, default=5,
        help="Max Hamming distance for dHash match (default: 5)",
    )
    proc.add_argument(
        "--simhash-threshold",
        type=float, default=0.95,
        help="Min SimHash cosine similarity (default: 0.95)",
    )
    proc.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "process":
        return _cmd_process(args)

    return 1


def _cmd_process(args: argparse.Namespace) -> int:
    """Execute the process command."""
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    source = Path(args.source)
    if not source.is_dir():
        print(f"Error: Not a directory: {source}", file=sys.stderr)
        return 1

    device = None if args.device == "auto" else args.device

    from .pipeline import run_pipeline

    try:
        state, report = run_pipeline(
            source_dir=args.source,
            window=args.window,
            min_cluster_size=args.min_cluster_size,
            quality_threshold=args.quality_threshold,
            top_k=args.top_k,
            clip_device=device,
            clip_batch_size=args.batch_size,
            output_path=args.output or None,
            dhash_threshold=args.dhash_threshold,
            simhash_threshold=args.simhash_threshold,
        )
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        logging.exception("Pipeline failed")
        return 1

    # Print summary to stdout
    print(f"\n{'='*60}")
    print(f"  SnapTidy Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Total photos scanned : {report['total_photos']}")
    print(f"  After deduplication  : {report['after_dedup']}")
    print(f"  Duplicate groups     : {len(state.dedup_report)}")
    print(f"  Final groups         : {report.get('n_groups', '?')}")
    print(f"  Keep suggestions     : {report.get('summary', {}).get('keep', 0)}")
    print(f"  Archive suggestions  : {report.get('summary', {}).get('archive', 0)}")
    print(f"  Elapsed              : {state.elapsed_ms/1000:.1f}s")
    print(f"{'='*60}")

    if args.output:
        print(f"  Report saved to      : {args.output}")
    else:
        from .export import report_to_string
        print(f"  Full report (stdout)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
