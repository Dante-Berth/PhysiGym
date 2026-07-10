#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
import shutil
from multiprocessing import Pool, cpu_count
import re
import argparse

import numpy as np

BASE_DIR = Path("/home/alex/Physi/PhysiCell/data")

# _render_frame lives in the physigym wrapper; make it importable so this script
# can turn a frames.npz (raw per-step arrays) into rendered PNGs.
_ENVS_DIR = (
    Path(__file__).resolve().parent
    / "custom_modules"
    / "physigym"
    / "physigym"
    / "envs"
)
if str(_ENVS_DIR) not in sys.path:
    sys.path.insert(0, str(_ENVS_DIR))


def _render_npz_to_frames(npz_path: Path, frames_dir: Path):
    """Load a frames.npz and render each step to frames_dir/frame_%05d.png."""
    from wrapper import _render_frame  # imported lazily (matplotlib is heavy)
    import cv2

    data = np.load(npz_path, allow_pickle=True)
    cells = data["cells"]
    subs = data["subs"]
    actions = data["actions"]
    rewards = data["rewards"]
    doses = data["doses"]
    # physical injection geometry as applied to the sim (may be absent in
    # older npz files → fall back to None so _render_frame re-derives).
    inj_x = data["inj_x"] if "inj_x" in data.files else None
    inj_y = data["inj_y"] if "inj_y" in data.files else None
    inj_r = data["inj_radius"] if "inj_radius" in data.files else None

    cell_type_names = list(data["cell_type_names"])
    cell_type_colors = {
        n: c for n, c in zip(cell_type_names, list(data["cell_type_colors"]))
    }
    substrate_names = list(data["substrate_names"])
    type_mode = str(data["type_mode"])
    episode = int(data["episode"])
    x_min, x_max = float(data["x_min"]), float(data["x_max"])
    y_min, y_max = float(data["y_min"]), float(data["y_max"])

    frames_dir.mkdir(exist_ok=True)
    for i in range(cells.shape[0]):
        rgb = _render_frame(
            cells_img=cells[i],
            subs_img=subs[i],
            cell_type_names=cell_type_names,
            cell_type_colors=cell_type_colors,
            substrate_names=substrate_names,
            action=actions[i],
            step=i,
            episode=episode,
            type_mode=type_mode,
            reward_history=list(rewards[: i + 1]),
            dose_history=list(doses[: i + 1]),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            inj_x=None if inj_x is None else float(inj_x[i]),
            inj_y=None if inj_y is None else float(inj_y[i]),
            inj_radius=None if inj_r is None else float(inj_r[i]),
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frames_dir / f"frame_{i:05d}.png"), bgr)


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

    Modes:
    1. If frames.npz exists → render PNGs, then ffmpeg → video.mp4
    2. If frames/ exists with frame_*.png files → ffmpeg directly to video.mp4
    3. If snapshot*.svg files exist → convert SVGs → ffmpeg
    4. If none → cleanup only
    """
    # Preferred path: raw frame data dumped by the wrapper (frames.npz). Render
    # the PNGs here (heavy matplotlib work, off the simulation's critical path).
    npz_path = run_dir / "frames.npz"
    frames_dir = run_dir / "frames"
    if npz_path.exists():
        try:
            _render_npz_to_frames(npz_path, frames_dir)
            video_file = run_dir / "video.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "10",
                    "-i",
                    str(frames_dir / "frame_%05d.png"),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    str(video_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            shutil.rmtree(frames_dir, ignore_errors=True)
            npz_path.unlink()
            _cleanup_non_essential(run_dir)
            print(f"Processed {run_dir}: video created from frames.npz")
            return
        except Exception as e:
            print(f"\n[ERROR] failed rendering {npz_path}: {e}")
            shutil.rmtree(frames_dir, ignore_errors=True)
            return

    # Fallback: pre-rendered PNG frames already on disk.
    if frames_dir.exists() and list(frames_dir.glob("frame_*.png")):
        try:
            video_file = run_dir / "video.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "10",
                    "-i",
                    str(frames_dir / "frame_%05d.png"),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    str(video_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
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
                subprocess.run(
                    [
                        "rsvg-convert",
                        "-w",
                        target_width,
                        "-o",
                        str(png_file),
                        str(svg_file),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    [
                        "inkscape",
                        str(svg_file),
                        "--export-width",
                        target_width,
                        "--export-filename",
                        str(png_file),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        video_file = run_dir / "video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "10",
                "-i",
                str(tmp_dir / "frame_%05d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                str(video_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

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


def _run_number(run_dir: Path) -> int:
    """Extract the numeric index from a run_XXXXXX directory name (0 if absent)."""
    m = re.search(r"run_0*(\d+)", run_dir.name)
    return int(m.group(1)) if m else 0


def find_all_runs(base_dir: Path, last_n: int = None):
    """
    Recursively find run_* directories anywhere under base_dir.

    If last_n is given, keep only the last_n highest-numbered runs *per
    episodes group* — i.e. per (env, train/test) directory — so each env
    keeps its most recent runs instead of rendering the whole history.
    Runs not selected are returned separately so the caller can drop their
    raw frames.npz rather than leaving stale data behind.

    Returns (selected_runs, skipped_runs).
    """
    all_runs = [d for d in base_dir.rglob("run_*") if d.is_dir()]
    if last_n is None:
        return all_runs, []

    # group by parent (the episodes/ dir, unique per env + split)
    groups: dict[Path, list[Path]] = {}
    for d in all_runs:
        groups.setdefault(d.parent, []).append(d)

    selected, skipped = [], []
    for parent, runs in groups.items():
        runs_sorted = sorted(runs, key=_run_number)
        keep = runs_sorted[-last_n:] if last_n > 0 else []
        drop = runs_sorted[:-last_n] if last_n > 0 else runs_sorted
        selected.extend(keep)
        skipped.extend(drop)
    return selected, skipped


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
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Only render videos for the last N runs per env (per episodes/ "
        "group). Older runs are skipped and their raw frames.npz removed. "
        "Default: render all runs.",
    )
    args = parser.parse_args()

    runs, skipped = find_all_runs(args.base_dir, last_n=args.last_n)
    print(
        f"Found {len(runs) + len(skipped)} runs in {args.base_dir}; "
        f"processing {len(runs)}, skipping {len(skipped)}"
    )

    # For skipped runs, drop the heavy raw frames.npz so it isn't kept around
    # as if a render had failed. Leave any already-rendered video.mp4 intact.
    for d in skipped:
        npz = d / "frames.npz"
        if npz.exists():
            npz.unlink()

    with Pool(args.workers) as pool:
        pool.map(process_run, runs)

    if not args.no_cleanup:
        print("Running final cleanup sweep...")
        # Keep .npz too: a surviving frames.npz means its render failed, so leave
        # it in place for a rerun rather than silently discarding the raw data.
        KEEP = {".mp4", ".csv", ".npz"}
        deleted = 0
        for f in args.base_dir.rglob("*"):
            if f.is_file() and f.suffix not in KEEP:
                f.unlink()
                deleted += 1
        print(f"Final sweep deleted {deleted} files.")

    print("All done!")


if __name__ == "__main__":
    main()
