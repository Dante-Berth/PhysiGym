"""
plot_curves_from_data.py
────────────────────────
Generate all figures for the EWRL 2026 paper directly from episode CSVs
in PhysiCell/data/.

Outputs (to paper/ewrl_2026_physigym/img/):
  train_return_mean50.pdf/png  — training return vs env steps, mean ± std
  test_return_mean50.pdf/png   — test return vs env steps, mean ± std
  test_std.pdf/png             — std of test return vs env steps
  fig_episode_comparison.pdf/png   — run_000143 seed 64  (I2 / I1 / S3)
  fig_episode_comparison_2.pdf/png — run_000147 seed 128 (I2 / I1 / S3)

Usage:
    python plot_curves_from_data.py \
        --data_dir PhysiCell/data \
        --out_dir  PhysiGym/paper/ewrl_2026_physigym/img
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent.parent / "PhysiCell" / "data"
DEFAULT_OUT  = SCRIPT_DIR.parent / "paper" / "ewrl_2026_physigym" / "img"

# ── style ─────────────────────────────────────────────────────────────────────
MODE_META = {
    "img_mc_cells_substrates": dict(
        label="I2 — img\_mc\_cells\_substrates",
        short="I2", color="#e63946", ls="-",  lw=2.0, z=5),
    "img_mc_cells": dict(
        label="I1 — img\_mc\_cells",
        short="I1", color="#2a9d8f", ls="-",  lw=1.8, z=4),
    "spatial_scalars_cells": dict(
        label="S3 — spatial\_scalars\_cells",
        short="S3", color="#4361ee", ls="--", lw=1.6, z=3),
    "spatial_scalars_cells_spatial_substrates": dict(
        label="S5 — spatial\_scalars\_cells\_spatial\_substrates",
        short="S5", color="#57cc99", ls=":",  lw=1.4, z=2),
    "scalars_cells_substrates": dict(
        label="S2 — scalars\_cells\_substrates",
        short="S2", color="#f4a261", ls="-.", lw=1.4, z=2),
    "scalars_cells": dict(
        label="S1 — scalars\_cells",
        short="S1", color="#adb5bd", ls="--", lw=1.2, z=1),
}
MODE_ORDER = list(MODE_META.keys())

# Regex to parse folder name: TME_V2_<seed>_<mode_key>_<timestamp>
_RUN_RE = re.compile(r"^TME_V2_(\d+)_(.+?)_(\d+)$")


def _fig_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.framealpha": 0.9,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })


def _save(out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"{name}.{ext}")
    print(f"  saved → {out_dir}/{name}.[pdf|png]")
    plt.close()


def _ewma(arr: np.ndarray, span: int = 50) -> np.ndarray:
    a = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


def _infer_mode(folder: str):
    """Return (seed, mode_key) from folder name, or (None, None)."""
    m = _RUN_RE.match(folder)
    if not m:
        return None, None
    seed, obs = int(m.group(1)), m.group(2)
    for key in sorted(MODE_META, key=len, reverse=True):
        if obs == key or obs.startswith(key + "_") or obs.startswith(key):
            return seed, key
    return seed, None


# ── data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path, mode_key: str):
    """
    Load all train episodes for one run directory.
    Returns dict: {env_id: [(cum_global_steps, episode_return), ...]}
    where cum_global_steps is accumulated across ALL envs in this run
    (proxy for total env-steps seen by the learner).
    """
    # Gather (run_num, env_id, n_steps, return) tuples
    records = []
    for env_dir in sorted(run_dir.iterdir()):
        if not env_dir.is_dir() or not env_dir.name.startswith("env"):
            continue
        env_id = int(env_dir.name[3:])
        ep_root = env_dir / "train" / "episodes"
        if not ep_root.exists():
            continue
        for ep_dir in sorted(ep_root.iterdir()):
            if not ep_dir.is_dir():
                continue
            csv = ep_dir / "data.csv"
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if df.empty or "cumulative_reward" not in df.columns:
                continue
            run_num = int(ep_dir.name.split("_")[1])
            ep_ret  = float(df["cumulative_reward"].iloc[-1])
            n_steps = len(df)
            records.append((run_num, env_id, n_steps, ep_ret))

    if not records:
        return []

    records.sort(key=lambda x: (x[0], x[1]))   # sort by (run_num, env_id)

    # Compute cumulative global steps = sum of all steps across all envs
    cum = 0
    out = []  # [(cum_steps, episode_return)]
    for _, _, n_steps, ep_ret in records:
        cum += n_steps
        out.append((cum, ep_ret))
    return out


def load_all_runs(data_dir: Path):
    """
    Returns {mode_key: {seed: [(cum_steps, ep_return), ...]}}
    Only train episodes; each seed may have one or more run dirs (same mode, diff timestamp).
    """
    results = defaultdict(lambda: defaultdict(list))

    for run_dir in sorted(data_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "comparisons":
            continue
        seed, mode_key = _infer_mode(run_dir.name)
        if mode_key is None or mode_key not in MODE_META:
            continue

        episodes = load_run(run_dir, mode_key)
        if episodes:
            # If there are multiple run dirs for the same (seed, mode), append
            # (they'll be averaged later if needed — here we just take the largest)
            existing = results[mode_key][seed]
            if not existing or len(episodes) > len(existing):
                results[mode_key][seed] = episodes

    return results


def load_test_returns_by_layout(data_dir: Path):
    """
    Returns {layout: {mode_key: {seed: [(cum_steps, ep_return), ...]}}}
    layout in ("all", "rect", "circ", "network_field")

    NOTE: rect/circ episodes are saved in the train/ folder (the wrapper
    routes them there based on episode numbering).  network_field test
    episodes live in test/.  We scan BOTH folders and split by type_mode.
    """
    from collections import defaultdict as _dd

    raw = _dd(lambda: _dd(lambda: _dd(list)))

    for run_dir in sorted(data_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "comparisons":
            continue
        seed, mode_key = _infer_mode(run_dir.name)
        if mode_key is None or mode_key not in MODE_META:
            continue

        # ── step 1: cumulative env steps per train run_num ─────────────────
        # We count network_field train episodes only (type_mode == network_field)
        # to build the x-axis, ignoring rect/circ interspersed episodes.
        steps_per_run: dict[int, int] = defaultdict(int)
        for env_dir in run_dir.iterdir():
            if not env_dir.is_dir() or not env_dir.name.startswith("env"):
                continue
            ep_root = env_dir / "train" / "episodes"
            if not ep_root.exists():
                continue
            for ep_dir in ep_root.iterdir():
                if not ep_dir.is_dir():
                    continue
                csv = ep_dir / "data.csv"
                if not csv.exists():
                    continue
                rn = int(ep_dir.name.split("_")[1])
                try:
                    n = sum(1 for _ in open(csv)) - 1
                except Exception:
                    n = 480
                steps_per_run[rn] += n

        if not steps_per_run:
            continue

        sorted_train_runs = sorted(steps_per_run.keys())
        cum = 0
        cum_after: dict[int, int] = {}
        for r in sorted_train_runs:
            cum += steps_per_run[r]
            cum_after[r] = cum

        def _steps_at(test_rn):
            cands = [r for r in sorted_train_runs if r < test_rn]
            return cum_after[max(cands)] if cands else 0

        # ── step 2: collect ALL episodes from both train/ and test/ ────────
        # rect/circ are in train/, network_field test episodes are in test/
        for env_dir in run_dir.iterdir():
            if not env_dir.is_dir() or not env_dir.name.startswith("env"):
                continue
            for split in ("train", "test"):
                ep_root = env_dir / split / "episodes"
                if not ep_root.exists():
                    continue
                for ep_dir in ep_root.iterdir():
                    if not ep_dir.is_dir():
                        continue
                    csv = ep_dir / "data.csv"
                    if not csv.exists():
                        continue
                    try:
                        df = pd.read_csv(csv)
                    except Exception:
                        continue
                    if df.empty or "cumulative_reward" not in df.columns:
                        continue
                    if "type_mode" not in df.columns:
                        continue
                    tm = df["type_mode"].iloc[0]
                    # only collect test-mode episodes (rect, circ, or network_field test)
                    if split == "train" and tm == "network_field":
                        continue   # skip training network_field episodes
                    rn     = int(ep_dir.name.split("_")[1])
                    ep_ret = float(df["cumulative_reward"].iloc[-1])
                    x      = _steps_at(rn)
                    layout = "rect" if tm == "rectangle" else (
                             "circ" if tm == "circular" else "network_field")
                    raw[mode_key][seed][layout].append((x, ep_ret))
                    raw[mode_key][seed]["all"].append((x, ep_ret))

    # ── step 3: reorganise to {layout: {mode_key: {seed: sorted_list}}} ────
    layouts = ("all", "rect", "circ", "network_field")
    out = {lay: defaultdict(dict) for lay in layouts}
    for mode_key in raw:
        for seed in raw[mode_key]:
            for lay in layouts:
                pts = sorted(raw[mode_key][seed].get(lay, []), key=lambda t: t[0])
                if pts:
                    out[lay][mode_key][seed] = pts
    return out


def load_test_returns(data_dir: Path):
    """
    Returns {mode_key: {seed: [(cum_steps_at_test, test_return), ...]}}

    X-axis = cumulative environment steps across ALL 9 parallel envs at the
    moment the test episode was recorded.

    Train/test run numbers interleave: trains on 1,2,4,5,6,8,… and tests on
    3,7,11,… (every 4th run).  For test run_num r the appropriate x value is
    the cumulative steps from all train runs with run_num < r.

    We first build a per-run_num step count (summed over all envs), then
    compute prefix sums so that cum_at[r] = total steps in all train runs
    with run_num < r.
    """
    results = defaultdict(lambda: defaultdict(list))

    for run_dir in sorted(data_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "comparisons":
            continue
        seed, mode_key = _infer_mode(run_dir.name)
        if mode_key is None or mode_key not in MODE_META:
            continue

        # ── step 1: count steps per train run_num, summed over all envs ──
        steps_per_run: dict[int, int] = defaultdict(int)
        for env_dir in run_dir.iterdir():
            if not env_dir.is_dir() or not env_dir.name.startswith("env"):
                continue
            ep_root = env_dir / "train" / "episodes"
            if not ep_root.exists():
                continue
            for ep_dir in ep_root.iterdir():
                if not ep_dir.is_dir():
                    continue
                csv = ep_dir / "data.csv"
                if not csv.exists():
                    continue
                run_num = int(ep_dir.name.split("_")[1])
                try:
                    n = sum(1 for _ in open(csv)) - 1
                except Exception:
                    n = 480
                steps_per_run[run_num] += n

        if not steps_per_run:
            continue

        # ── step 2: prefix sum — cum_at[r] = total steps before train run r ──
        # i.e. cumulative steps from all train run_nums strictly < r
        sorted_train_runs = sorted(steps_per_run.keys())
        cum = 0
        cum_after: dict[int, int] = {}   # cum_after[r] = steps AFTER run r completes
        for r in sorted_train_runs:
            cum += steps_per_run[r]
            cum_after[r] = cum

        def _steps_at_test(test_rn: int) -> int:
            """Cumulative steps when test run test_rn was triggered."""
            # Find the largest train run_num strictly less than test_rn
            candidates = [r for r in sorted_train_runs if r < test_rn]
            if not candidates:
                return 0
            return cum_after[max(candidates)]

        # ── step 3: collect test episode returns with correct x ────────────
        for env_dir in run_dir.iterdir():
            if not env_dir.is_dir() or not env_dir.name.startswith("env"):
                continue
            ep_root = env_dir / "test" / "episodes"
            if not ep_root.exists():
                continue
            for ep_dir in ep_root.iterdir():
                if not ep_dir.is_dir():
                    continue
                csv = ep_dir / "data.csv"
                if not csv.exists():
                    continue
                try:
                    df = pd.read_csv(csv)
                except Exception:
                    continue
                if df.empty or "cumulative_reward" not in df.columns:
                    continue
                run_num = int(ep_dir.name.split("_")[1])
                ep_ret  = float(df["cumulative_reward"].iloc[-1])
                x       = _steps_at_test(run_num)
                results[mode_key][seed].append((x, ep_ret))

    # sort by x within each (mode, seed)
    for mk in results:
        for sd in results[mk]:
            results[mk][sd].sort(key=lambda t: t[0])

    return results


# ── interpolate to common x-grid ──────────────────────────────────────────────

def _to_common_grid(seed_curves: list, n_points: int = 300, smooth: int = 50):
    """
    seed_curves: list of [(cum_steps, ep_return)] per seed.
    Returns (x_grid, mean, std) where x_grid is n_points evenly spaced.
    """
    if not seed_curves:
        return None, None, None

    x_max = max(t[-1][0] for t in seed_curves if t)
    x_grid = np.linspace(0, x_max, n_points)

    smoothed = []
    for curve in seed_curves:
        if not curve:
            continue
        xs = np.array([t[0] for t in curve], dtype=float)
        ys = np.array([t[1] for t in curve], dtype=float)
        # ewma smooth on the raw series first
        ys_s = _ewma(ys, smooth)
        # then interpolate to common grid
        smoothed.append(np.interp(x_grid, xs, ys_s))

    if not smoothed:
        return None, None, None

    mat  = np.array(smoothed)
    return x_grid, mat.mean(axis=0), mat.std(axis=0)


# ── learning curve figures ────────────────────────────────────────────────────

def plot_learning_curves(train_data, test_data, out_dir: Path, smooth: int = 50):
    """
    Two-panel figure: training return (left) + test return (right),
    both with mean ± std shading vs environment steps.
    Also saves individual PDFs for each panel.
    """
    _fig_style()

    def _draw(ax, data_dict, title, ylabel):
        for mode in MODE_ORDER:
            if mode not in data_dict:
                continue
            seed_curves = list(data_dict[mode].values())
            x, mean, std = _to_common_grid(seed_curves, smooth=smooth)
            if x is None:
                continue
            meta = MODE_META[mode]
            ax.plot(x / 1e5, mean, color=meta["color"], ls=meta["ls"],
                    lw=meta["lw"], label=meta["short"], zorder=meta["z"])
            ax.fill_between(x / 1e5, mean - std, mean + std,
                            color=meta["color"], alpha=0.18, zorder=meta["z"] - 1)
        ax.axhline(0, color="gray", lw=0.6, ls="--", zorder=0)
        ax.set_xlabel(r"Environment steps ($\times 10^5$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # ── combined 2-panel ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _draw(axes[0], train_data, "Training return (EWMA-50)", "Cumulative return")
    _draw(axes[1], test_data,  "Test return (EWMA-50)",    "Cumulative return")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=6,
                   fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    _save(out_dir, "learning_curves_combined")

    # ── individual panels ─────────────────────────────────────────────────
    for data_dict, fname, title, ylabel in [
        (train_data, "train_return_mean50", "Training return (EWMA-50, mean $\\pm$ std)",
         "Cumulative return (training)"),
        (test_data,  "test_return_mean50",  "Test return (EWMA-50, mean $\\pm$ std)",
         "Cumulative return (test)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        _draw(ax, data_dict, title, ylabel)
        ax.legend(fontsize=7.5, loc="upper left")
        _save(out_dir, fname)


def plot_layout_curves(layout_data: dict, out_dir: Path, smooth: int = 50):
    """
    Two-panel figure: rectangle return (left) + circular return (right),
    mean ± std vs environment steps (×10⁵).
    """
    _fig_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, layout_key, title in [
        (axes[0], "rect", "Test return — rectangle layout (mean $\\pm$ std)"),
        (axes[1], "circ", "Test return — circular layout (mean $\\pm$ std)"),
    ]:
        data_dict = layout_data.get(layout_key, {})
        for mode in MODE_ORDER:
            if mode not in data_dict:
                continue
            seed_curves = list(data_dict[mode].values())
            x, mean, std = _to_common_grid(seed_curves, smooth=smooth)
            if x is None:
                continue
            meta = MODE_META[mode]
            ax.plot(x / 1e5, mean, color=meta["color"], ls=meta["ls"],
                    lw=meta["lw"], label=meta["short"], zorder=meta["z"])
            ax.fill_between(x / 1e5, mean - std, mean + std,
                            color=meta["color"], alpha=0.18, zorder=meta["z"] - 1)
        ax.axhline(0, color="gray", lw=0.6, ls="--", zorder=0)
        ax.set_xlabel(r"Environment steps ($\times 10^5$)")
        ax.set_ylabel("Cumulative return")
        ax.set_title(title)
        ax.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    _save(out_dir, "test_layout_curves")

    # individual panels
    for layout_key, fname, title in [
        ("rect", "test_layout_rect", "Test return — rectangle layout (mean $\\pm$ std)"),
        ("circ", "test_layout_circ", "Test return — circular layout (mean $\\pm$ std)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        data_dict = layout_data.get(layout_key, {})
        for mode in MODE_ORDER:
            if mode not in data_dict:
                continue
            seed_curves = list(data_dict[mode].values())
            x, mean, std = _to_common_grid(seed_curves, smooth=smooth)
            if x is None:
                continue
            meta = MODE_META[mode]
            ax.plot(x / 1e5, mean, color=meta["color"], ls=meta["ls"],
                    lw=meta["lw"], label=meta["short"], zorder=meta["z"])
            ax.fill_between(x / 1e5, mean - std, mean + std,
                            color=meta["color"], alpha=0.18, zorder=meta["z"] - 1)
        ax.axhline(0, color="gray", lw=0.6, ls="--", zorder=0)
        ax.set_xlabel(r"Environment steps ($\times 10^5$)")
        ax.set_ylabel("Cumulative return")
        ax.set_title(title)
        ax.legend(fontsize=7.5, loc="upper left")
        _save(out_dir, fname)


def plot_std_curves(train_data, test_data, out_dir: Path, smooth: int = 50):
    """Plot std across seeds for train and test (shows robustness)."""
    _fig_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for ax, data_dict, title in [
        (axes[0], train_data, "Std of training return across seeds"),
        (axes[1], test_data,  "Std of test return across seeds"),
    ]:
        for mode in MODE_ORDER:
            if mode not in data_dict:
                continue
            seed_curves = list(data_dict[mode].values())
            if len(seed_curves) < 2:
                continue
            x, _, std = _to_common_grid(seed_curves, smooth=smooth)
            if x is None:
                continue
            meta = MODE_META[mode]
            ax.plot(x / 1e5, std, color=meta["color"], ls=meta["ls"],
                    lw=meta["lw"], label=meta["short"], zorder=meta["z"])
        ax.set_xlabel(r"Environment steps ($\times 10^5$)")
        ax.set_ylabel("Std of cumulative return")
        ax.set_title(title)
        ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    _save(out_dir, "return_std")


# ── episode comparison figures ────────────────────────────────────────────────

# Exact CSV paths (verified to match report numbers)
EPISODE_SPECS = {
    "run_000143_seed64": {
        "title": "Episode comparison — run\\_000143 (seed 64, network-field)",
        "fname": "fig_episode_comparison",
        "I2": "TME_V2_64_img_mc_cells_substrates_1778446547/env4/test/episodes/run_000143/data.csv",
        "I1": "TME_V2_64_img_mc_cells_1778485079/env4/test/episodes/run_000143/data.csv",
        "S3": "TME_V2_64_spatial_scalars_cells_1778933152/env3/test/episodes/run_000143/data.csv",
    },
    "run_000147_seed128": {
        "title": "Episode comparison — run\\_000147 (seed 128, network-field)",
        "fname": "fig_episode_comparison_2",
        "I2": "TME_V2_128_img_mc_cells_substrates_1778523222/env3/test/episodes/run_000147/data.csv",
        "I1": "TME_V2_128_img_mc_cells_1778561927/env1/test/episodes/run_000147/data.csv",
        "S3": "TME_V2_128_spatial_scalars_cells_1778971599/env1/test/episodes/run_000147/data.csv",
    },
}

EPISODE_COLORS = {"I2": "#e63946", "I1": "#2a9d8f", "S3": "#4361ee"}
EPISODE_LS     = {"I2": "-",       "I1": "-",       "S3": "--"}


def plot_episode_comparison(data_dir: Path, out_dir: Path, spec_key: str):
    spec = EPISODE_SPECS[spec_key]
    _fig_style()

    fig = plt.figure(figsize=(8, 7.5))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)
    ax_ret  = fig.add_subplot(gs[0])
    ax_dose = fig.add_subplot(gs[1])
    ax_step = fig.add_subplot(gs[2])

    found_any = False
    stats = {}

    for label in ("I2", "I1", "S3"):
        csv = data_dir / spec[label]
        if not csv.exists():
            print(f"  WARNING: {label} not found at {csv}")
            continue
        df = pd.read_csv(csv)
        found_any = True

        steps   = df["step"].values
        cum_r   = df["cumulative_reward"].values
        cum_d   = df["cumulative_dose_spent"].values if "cumulative_dose_spent" in df else np.zeros(len(df))
        dose_s  = df["dose_spent"].values if "dose_spent" in df else np.zeros(len(df))
        n_tumor = df["number_tumor"].iloc[-1] if "number_tumor" in df.columns else "?"

        stats[label] = {
            "ret": float(df["cumulative_reward"].iloc[-1]),
            "dose": float(cum_d[-1]),
            "n_tumor": int(n_tumor) if n_tumor != "?" else "?",
        }

        c = EPISODE_COLORS[label]
        ls = EPISODE_LS[label]
        ax_ret.plot(steps, cum_r,  color=c, ls=ls, lw=1.8,
                    label=f"{label}  (ret={stats[label]['ret']:+.0f}, tumor={stats[label]['n_tumor']})")
        ax_dose.plot(steps, cum_d, color=c, ls=ls, lw=1.8)
        ax_step.plot(steps, dose_s, color=c, ls=ls, lw=1.0, alpha=0.85)

    for ax in (ax_ret, ax_dose, ax_step):
        ax.axhline(0, color="gray", lw=0.6, ls="--", zorder=0)

    ax_ret.set_ylabel("Cumulative return")
    ax_dose.set_ylabel("Cumulative dose")
    ax_step.set_ylabel("Dose per step")
    ax_step.set_xlabel("Episode step")

    ax_ret.set_title(spec["title"])
    ax_ret.legend(fontsize=8, loc="upper left")

    if not found_any:
        fig.text(0.5, 0.5, "Episode CSVs not found", ha="center",
                 va="center", fontsize=12, color="gray")

    _save(out_dir, spec["fname"])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT))
    parser.add_argument("--smooth",   type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.exists():
        sys.exit(f"ERROR: data_dir not found: {data_dir}")

    print(f"Loading train episodes from {data_dir} …")
    train_data = load_all_runs(data_dir)
    print(f"  modes: {[k for k in train_data]}")

    print("Loading test episodes (all + by layout) …")
    layout_data = load_test_returns_by_layout(data_dir)
    test_data   = layout_data.get("all", {})   # used for the main test curve

    print("Plotting learning curves (env-steps x-axis, mean±std) …")
    plot_learning_curves(train_data, test_data, out_dir, args.smooth)

    print("Plotting rect / circ layout curves …")
    plot_layout_curves(layout_data, out_dir, args.smooth)

    print("Plotting std curves …")
    plot_std_curves(train_data, test_data, out_dir, args.smooth)

    print("Plotting episode comparisons …")
    for spec_key in EPISODE_SPECS:
        plot_episode_comparison(data_dir, out_dir, spec_key)

    print("Done.")


if __name__ == "__main__":
    main()
