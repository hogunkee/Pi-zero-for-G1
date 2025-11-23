#!/usr/bin/env python3
"""
Rewrite timestamp columns in LeRobot-style Parquet episodes to match actual video frame timestamps.

This ensures perfect alignment between parquet timestamps and video frame PTS (Presentation Time Stamps).

Usage:
  python retime_parquet_from_video.py /data/lerobot --output-dir /data/lerobot_retimed
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False
    logging.warning("torchcodec not available, will try pyav")

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


def get_video_timestamps_torchcodec(video_path: Path) -> np.ndarray:
    """Extract actual frame timestamps from video using torchcodec."""
    decoder = VideoDecoder(str(video_path))
    metadata = decoder.metadata
    num_frames = metadata.num_frames

    # Get all frame timestamps
    timestamps = []
    for i in range(num_frames):
        frame_data = decoder.get_frames_at(indices=[i])
        pts = frame_data.pts_seconds[0].item()
        timestamps.append(pts)

    return np.array(timestamps, dtype=np.float64)


def get_video_timestamps_pyav(video_path: Path) -> np.ndarray:
    """Extract actual frame timestamps from video using pyav."""
    timestamps = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            # PTS in seconds
            pts = float(frame.pts * stream.time_base)
            timestamps.append(pts)

    return np.array(timestamps, dtype=np.float64)


def get_video_timestamps(video_path: Path) -> np.ndarray:
    """Extract actual frame timestamps from video."""
    if HAS_TORCHCODEC:
        return get_video_timestamps_torchcodec(video_path)
    elif HAS_PYAV:
        return get_video_timestamps_pyav(video_path)
    else:
        raise ImportError("Neither torchcodec nor pyav is available. Install one of them.")


def detect_timestamp_cols(df: pd.DataFrame) -> List[str]:
    """Detect timestamp columns in dataframe."""
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if "timestamp" in cl or cl.endswith("_ts"):
            cols.append(c)
    return cols


def find_video_for_episode(dataset_root: Path, episode_parquet: Path) -> Path:
    """Find the corresponding video file for an episode parquet file.

    LeRobot dataset structure:
    - data/
      - chunk-000/
        - episode_000000.parquet
        - episode_000001.parquet
    - videos/
      - chunk-000/
        - episode_000000.mp4
        - episode_000001.mp4
    """
    # Get relative path from dataset root
    rel_path = episode_parquet.relative_to(dataset_root)

    # Replace 'data' with 'videos' and '.parquet' with '.mp4'
    video_rel_path = str(rel_path).replace('/data/', '/videos/').replace('.parquet', '.mp4')
    video_path = dataset_root / video_rel_path

    if not video_path.exists():
        # Try alternative: videos might be in same structure but different extension
        video_path = episode_parquet.parent.parent.parent / 'videos' / episode_parquet.parent.name / f"{episode_parquet.stem}.mp4"

    return video_path


def process_episode(src_path: Path, dst_path: Path, dataset_root: Path, args, report: List) -> None:
    """Process a single episode parquet file."""
    try:
        df = pd.read_parquet(src_path)
    except Exception as e:
        report.append((str(src_path), f"ERROR(read failed: {e})", {}))
        return

    # Find corresponding video
    video_path = find_video_for_episode(dataset_root, src_path)
    if not video_path.exists():
        report.append((str(src_path), f"SKIP(video not found: {video_path})", {}))
        return

    # Get actual video timestamps
    try:
        video_timestamps = get_video_timestamps(video_path)
    except Exception as e:
        report.append((str(src_path), f"ERROR(video read failed: {e})", {}))
        return

    n_parquet = len(df)
    n_video = len(video_timestamps)

    if n_parquet != n_video:
        logging.warning(
            f"{src_path.name}: Frame count mismatch! "
            f"Parquet has {n_parquet} rows, video has {n_video} frames"
        )
        if n_parquet > n_video:
            report.append((str(src_path), f"ERROR(parquet has more frames than video: {n_parquet} > {n_video})", {}))
            return
        # If video has more frames, we can still use the first n_parquet frames
        video_timestamps = video_timestamps[:n_parquet]

    # Detect or use specified timestamp columns
    cols = args.columns if args.columns else detect_timestamp_cols(df)
    if not cols:
        report.append((str(src_path), "SKIP(no timestamp cols)", {}))
        return

    # Update timestamp columns
    stats = {}
    for col in cols:
        old_timestamps = df[col].to_numpy()

        # Compute statistics
        old_diffs = np.diff(old_timestamps)
        new_diffs = np.diff(video_timestamps)
        max_error = np.max(np.abs(old_timestamps - video_timestamps))

        stats[col] = {
            'max_error': float(max_error),
            'old_mean_delta': float(np.mean(old_diffs)) if len(old_diffs) > 0 else 0,
            'new_mean_delta': float(np.mean(new_diffs)) if len(new_diffs) > 0 else 0,
        }

        # Update with actual video timestamps
        df[col] = video_timestamps

        logging.info(
            f"{src_path.name} [{col}]: max_error={max_error:.4f}s, "
            f"old_Δ={stats[col]['old_mean_delta']:.4f}s, "
            f"new_Δ={stats[col]['new_mean_delta']:.4f}s"
        )

    if args.dry_run:
        report.append((str(src_path), "DRY_RUN(would update)", stats))
        return

    # Write updated parquet
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if args.inplace:
        if args.backup:
            bak = src_path.with_suffix(src_path.suffix + ".bak")
            if not bak.exists():
                import shutil
                shutil.copy2(src_path, bak)
        df.to_parquet(src_path, index=False)
    else:
        df.to_parquet(dst_path, index=False)

    report.append((str(src_path), "OK", stats))


def main():
    ap = argparse.ArgumentParser(
        description="Align parquet timestamps with actual video frame timestamps"
    )
    ap.add_argument("root", type=str, help="Dataset root containing data/ and videos/ directories")
    ap.add_argument("--columns", nargs="*", default=None,
                   help="Timestamp columns to rewrite; auto-detect if omitted")
    ap.add_argument("--pattern", default="**/data/**/*.parquet",
                   help="Glob pattern for parquet files (default: **/data/**/*.parquet)")
    ap.add_argument("--output-dir", type=str, default=None,
                   help="Destination root; if omitted uses <root>_retimed")
    ap.add_argument("--inplace", action="store_true",
                   help="Rewrite files in place")
    ap.add_argument("--backup", action="store_true",
                   help="Keep .bak when using --inplace")
    ap.add_argument("--dry-run", action="store_true",
                   help="Plan only, no writes")
    args = ap.parse_args()

    if not HAS_TORCHCODEC and not HAS_PYAV:
        print("[ERROR] Neither torchcodec nor pyav is available. Please install one:",
              file=sys.stderr)
        print("  pip install torchcodec  # Recommended", file=sys.stderr)
        print("  pip install av          # Alternative", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        sys.exit(1)

    if args.inplace:
        out_root = root
    elif args.output_dir is None:
        out_root = Path(str(root) + "_retimed").resolve()
    else:
        out_root = Path(args.output_dir).resolve()

    # Find all parquet files
    files = sorted(root.glob(args.pattern))
    if not files:
        print(f"[WARN] No files matched '{args.pattern}' under {root}", file=sys.stderr)
        sys.exit(0)

    print(f"\nFound {len(files)} parquet files to process")
    print(f"Using {'torchcodec' if HAS_TORCHCODEC else 'pyav'} for video decoding")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    report = []
    for i, src in enumerate(files, 1):
        rel = src.relative_to(root)
        dst = src if args.inplace else (out_root / rel)

        print(f"[{i}/{len(files)}] Processing {rel}...")
        process_episode(src, dst, root, args, report)

    # Summary
    success = [r for r in report if r[1] == "OK" or r[1].startswith("DRY_RUN")]
    errors = [r for r in report if r[1].startswith("ERROR")]
    skipped = [r for r in report if r[1].startswith("SKIP")]

    print(f"\n{'='*60}")
    print(f"Total: {len(files)} | Success: {len(success)} | Errors: {len(errors)} | Skipped: {len(skipped)}")

    if errors:
        print(f"\nErrors:")
        for path, status, _ in errors:
            print(f"  - {Path(path).name}: {status}")

    if success:
        print(f"\nSuccessfully processed files (showing first 10):")
        for path, status, stats in success[:10]:
            print(f"  - {Path(path).name}:")
            for col, col_stats in stats.items():
                print(f"      {col}: max_error={col_stats['max_error']:.4f}s")


if __name__ == "__main__":
    main()
