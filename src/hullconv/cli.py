"""Command-line helpers shared by the scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from hullconv.data import TERNARY_SYSTEMS, data_dir, output_dir


def parser(description: str, systems: tuple[str, ...] = TERNARY_SYSTEMS
           ) -> argparse.ArgumentParser:
    """Parser with --data, --out, --systems and --n-jobs."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default=None,
                   help="data directory (default: $HULLCONV_DATA or ./data)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--systems", nargs="+", default=list(systems),
                   help="system tags, e.g. LRO LHO")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="parallel workers (-1: all cores)")
    return p


def resolve(args: argparse.Namespace) -> tuple[Path, Path]:
    """Data and output directories of parsed arguments."""
    return data_dir(args.data), output_dir(args.out)
