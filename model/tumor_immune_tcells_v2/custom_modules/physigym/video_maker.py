#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import shutil
from multiprocessing import Pool, cpu_count
import re
import argparse

BASE_DIR = Path("/home/alex/Physi/PhysiCell/data")

# Check converter (only needed if processing SVGs, not for frame rendering)
converter = None
if shutil.which("rsvg-convert"):
    converter = "rsvg-convert"
elif shutil.which("inkscape"):
    converter = "inkscape"

def get_snapshot_number(f: Path):
    match = re.search(r"snapshot0*(\d+)\.svg", f.name)
    return int(match.group(1)) if match else 0

def process_run(run_dir: Path):
    """
    Process a run directory: either compile pre-rendered frames or convert SVGs to video.

    Two modes:
    1. If frames/ exists with frame_*.png files → ffmpeg directly to video.mp4
    2. If snapshot*.svg files exist → convert SVGs → ffmpeg
    3. If neither → cleanup only
    """
    # Check for pre-rendered frames (from PhysiCellModelWrapper._compile_video deferred path)
    frames_dir = run_dir / "frames"
    if frames_dir.exists() and list(frames_dir.glob("frame_*.png")):
        try:
            video_file = run_dir / "video.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "10",
                "-i", str(frames_dir / "frame_%05d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-threads", "1",
                str(video_file)
            ], check=True, capture_output=True, text=True)
            shutil.rmtree(frames_dir)
            _cleanup_non_essential(run_dir)
            print(f"Processed {run_dir}: video created from pre-rendered frames")
            return
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] ffmpeg failed in {run_dir}.")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Stderr:\n{e.stderr}")
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            return

    # Fall back to SVG conversion (legacy PhysiCell output)
    if converter is None:
        # No converter available and no pre-rendered frames
        if (run_dir / "video.mp4").exists():
            _cleanup_non_essential(run_dir)
        return

    svg_files = list(run_dir.glob("snapshot*.svg"))
    if not svg_files:
        if (run_dir / "video.mp4").exists():
            _cleanup_non_essential(run_dir)
        return

    svg_files.sort(key=get_snapshot_number)

    tmp_dir = run_dir / "tmp_png"
    tmp_dir.mkdir(exist_ok=True)

    try:
        target_width = "512"

        for i, svg_file in enumerate(svg_files):
            png_file = tmp_dir / f"frame_{i:05d}.png"
            if converter == "rsvg-convert":
                subprocess.run([
                    "rsvg-convert",
                    "-w", target_width,
                    "-o", str(png_file),
                    str(svg_file)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run([
                    "inkscape",
                    str(svg_file),
                    "--export-width", target_width,
                    "--export-filename", str(png_file)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        video_file = run_dir / "video.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "10",
            "-i", str(tmp_dir / "frame_%05d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-threads", "1",
            str(video_file)
        ], check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Process failed in {run_dir}.")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stderr:\n{e.stderr}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return

    # Cleanup tmp PNGs
    shutil.rmtree(tmp_dir)

    # Cleanup SVGs
    for svg_file in svg_files:
        svg_file.unlink()

    # Cleanup .xml and .mat — keep only .mp4 and .csv
    _cleanup_non_essential(run_dir)

    print(f"Processed {run_dir}: video created from SVG snapshots")


def _cleanup_non_essential(run_dir: Path):
    """Delete all files except .mp4 and .csv in a run directory."""
    KEEP = {".mp4", ".csv"}
    for f in run_dir.iterdir():
        if f.is_file() and f.suffix not in KEEP:
            f.unlink()


def find_all_runs(base_dir: Path):
    """
    Recursively find ALL run_* directories anywhere under base_dir.
    Works regardless of nesting depth.
    """
    return [d for d in base_dir.rglob("run_*") if d.is_dir()]


def main():
    parser = argparse.ArgumentParser(
        description="Compile videos from PhysiCell simulation runs. Handles both legacy SVG "
                    "snapshots and pre-rendered frames from PhysiCellModelWrapper."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help=f"Base data directory (default: {BASE_DIR})",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup of non-essential files after video compilation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of worker processes (default: CPU_COUNT // 2)",
    )
    args = parser.parse_args()

    runs = find_all_runs(args.base_dir)
    print(f"Found {len(runs)} runs to process in {args.base_dir}")

    with Pool(args.workers) as pool:
        pool.map(process_run, runs)

    if not args.no_cleanup:
        print("Running final cleanup sweep...")
        KEEP = {".mp4", ".csv"}
        deleted = 0
        for f in args.base_dir.rglob("*"):
            if f.is_file() and f.suffix not in KEEP:
                f.unlink()
                deleted += 1
        print(f"Final sweep deleted {deleted} files.")

    print("All done!")

if __name__ == "__main__":
    main()