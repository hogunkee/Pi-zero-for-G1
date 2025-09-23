# -*- coding: utf-8 -*-
"""
Rewrite timestamp columns in LeRobot-style Parquet episodes to a uniform FPS timeline.

Usage:
  python retime_parquet_timestamps.py /data/lerobot --fps 20 \
      --unit s --anchor zero --output-dir /data/lerobot_retimed
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

def unit_scale(unit: str) -> float:
    return {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]

def detect_timestamp_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if "timestamp" in cl or cl.endswith("_ts"):
            cols.append(c)
    return cols

def make_uniform(n: int, fps: float, unit: str, base=0.0, as_int=False):
    t = (np.arange(n, dtype=np.float64) / float(fps)) * unit_scale(unit)
    t += float(base)
    if as_int:
        t = np.rint(t).astype(np.int64)
    return t

def verify_monotonic(ts) -> dict:
    ts = np.asarray(ts, dtype=np.float64)
    diffs = np.diff(ts)
    if len(diffs) == 0:
        return {"len": len(ts), "min_step": float("nan"), "median_step": float("nan"), "neg_jumps": 0, "zero_steps": 0}
    return {
        "len": len(ts),
        "min_step": float(np.min(diffs)),
        "median_step": float(np.median(diffs)),
        "neg_jumps": int(np.sum(diffs < 0)),
        "zero_steps": int(np.sum(diffs == 0)),
    }

def process_file(src_path: Path, dst_path: Path, args, report: list):
    df = pd.read_parquet(src_path)
    cols = args.columns if args.columns else detect_timestamp_cols(df)
    if not cols:
        report.append((str(src_path), "SKIP(no timestamp cols)", {}))
        return
    n = len(df)
    for col in cols:
        base = float(df[col].iloc[0]) if args.anchor == "first" else 0.0
        df[col] = make_uniform(n, args.fps, args.unit, base=base, as_int=args.as_int)
        stats = verify_monotonic(df[col].to_numpy())
        report.append((str(src_path), f"OK({col})", stats))

    if args.dry_run:
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if args.inplace:
        if args.backup:
            bak = src_path.with_suffix(src_path.suffix + ".bak")
            if not bak.exists():
                src_path.replace(bak)
        df.to_parquet(src_path, index=False)
    else:
        df.to_parquet(dst_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Dataset root (search recursively for *.parquet)")
    ap.add_argument("--fps", type=float, required=True, help="Target frames-per-second")
    ap.add_argument("--columns", nargs="*", default=None, help="Timestamp columns to rewrite; auto-detect if omitted")
    ap.add_argument("--unit", choices=["s","ms","us","ns"], default="s", help="Output time unit")
    ap.add_argument("--anchor", choices=["zero","first"], default="zero", help="Start at 0 or keep first value as base")
    ap.add_argument("--int", dest="as_int", action="store_true", help="Store as int64 instead of float64")
    ap.add_argument("--pattern", default="*.parquet", help="Glob pattern, recursive")
    ap.add_argument("--output-dir", type=str, default=None, help="Destination root; if omitted uses <root>_retimed")
    ap.add_argument("--inplace", action="store_true", help="Rewrite files in place")
    ap.add_argument("--backup", action="store_true", help="Keep .bak when using --inplace")
    ap.add_argument("--dry-run", action="store_true", help="Plan only, no writes")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERR] root not found: {root}", file=sys.stderr); sys.exit(1)
    out_root = root if args.inplace else Path(args.output_dir or (str(root) + "_retimed")).resolve()

    files = sorted(root.rglob(args.pattern))
    if not files:
        print(f"[WARN] No files matched '{args.pattern}' under {root}", file=sys.stderr); sys.exit(0)

    report = []
    for src in files:
        rel = src.relative_to(root)
        dst = src if args.inplace else (out_root / rel)
        process_file(src, dst, args, report)

    touched = [r for r in report if r[1].startswith("OK")]
    skipped = [r for r in report if r[1].startswith("SKIP")]
    print(f"\nProcessed: {len(files)}  |  Updated: {len(touched)}  |  Skipped: {len(skipped)}")
    for (path, status, stats) in touched[:5]:
        print(f" - {Path(path).name}: {status}, len={stats.get('len')}, "
              f"Δmin={stats.get('min_step'):.6f}, Δmedian={stats.get('median_step'):.6f}, "
              f"neg={stats.get('neg_jumps')}, zero={stats.get('zero_steps')}")
if __name__ == "__main__":
    main()
