#!/usr/bin/env python3
"""Extract and sort all zl/zs values from a lens FITS table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.table import Table


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Read a FITS table, extract zl/zs, and build a sorted redshift list."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "data" / "206_SL01_minimal.fits",
        help="Input FITS table.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=script_dir / "output" / "206_SL01_sorted_redshifts",
        help="Output file prefix. The script writes .json, .npy, and .txt files.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate values instead of returning unique sorted redshifts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    table = Table.read(args.input)
    required = {"zl", "zs"}
    missing = required - set(table.colnames)
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {sorted(missing)}")

    zl = np.asarray(table["zl"], dtype=float)
    zs = np.asarray(table["zs"], dtype=float)
    keep_mask = (zl <= 2.5) & (zs <= 2.5)
    excluded_count = int((~keep_mask).sum())

    zl = zl[keep_mask]
    zs = zs[keep_mask]
    combined = np.concatenate([zl, zs])

    if args.keep_duplicates:
        sorted_redshifts = np.sort(combined)
    else:
        sorted_redshifts = np.unique(np.sort(combined))

    values = sorted_redshifts.tolist()
    json_path = args.output_prefix.with_suffix(".json")
    npy_path = args.output_prefix.with_suffix(".npy")
    txt_path = args.output_prefix.with_suffix(".txt")

    json_path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    np.save(npy_path, sorted_redshifts)
    np.savetxt(txt_path, sorted_redshifts, fmt="%.10f")

    print(f"Input rows: {len(table)}")
    print(f"Excluded systems with zl > 2.5 or zs > 2.5: {excluded_count}")
    print(f"Remaining systems: {keep_mask.sum()}")
    print(f"Total zl+zs values: {combined.size}")
    print(f"Sorted list length: {len(values)}")
    print(f"Saved JSON list to: {json_path}")
    print(f"Saved NPY array to: {npy_path}")
    print(f"Saved TXT list to: {txt_path}")
    print("First 10 values:", values[:10])
    print("Last 10 values:", values[-10:])


if __name__ == "__main__":
    main()
