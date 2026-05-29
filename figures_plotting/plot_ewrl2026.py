"""
plot_ewrl2026.py
────────────────
Generates publication-quality figures for the EWRL 2026 paper:
  "State Representation Matters: Comparing Observation Spaces for
   RL-Driven Drug Delivery in a Tumor Microenvironment Simulation"

Two modes:
  1. FROM W&B CSV EXPORTS (recommended):
       Run download_all_wandb_histories.py first to populate
       wandb_csv_exports/<run_name>/history.csv, then call
           python plot_ewrl2026.py --source wandb --wandb_dir wandb_csv_exports
       Each history.csv must have columns:
           step, train_return_mean50, test_return_mean50,
           test_return_rectangle, test_return_circular
       (These are the W&B metric keys — adjust RUN_KEYS below if names differ.)

  2. FROM HARD-CODED PER-SEED SUMMARY (fallback, no CSV needed):
           python plot_ewrl2026.py --source static
       Uses the per-seed table from report_sac_tme_state_spaces.md.
       Produces bar charts + shaded final-value summary; no learning curves.

Output: ../paper/ewrl_2026_physigym/img/  (relative to this script)
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ── output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR.parent / "paper" / "ewrl_2026_physigym" / "img"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour / style palette per observation mode ───────────────────────────────
MODE_META = {
    "img_mc_cells_substrates": dict(
        label="I2 — img_mc_cells_substrates",
        color="#e63946",   # red
        ls="-",
        lw=2.0,
        zorder=5,
    ),
    "img_mc_cells": dict(
        label="I1 — img_mc_cells",
        color="#2a9d8f",   # teal
        ls="-",
        lw=1.8,
        zorder=4,
    ),
    "spatial_scalars_cells": dict(
        label="S3 — spatial_scalars_cells",
        color="#4361ee",   # blue
        ls="--",
        lw=1.6,
        zorder=3,
    ),
    "spatial_scalars_cells_spatial_substrates": dict(
        label="S5 — spatial_scalars_cells_spatial_substrates",
        color="#57cc99",   # light green
        ls=":",
        lw=1.4,
        zorder=2,
    ),
    "scalars_cells_substrates": dict(
        label="S2 — scalars_cells_substrates",
        color="#f4a261",   # orange
        ls="-.",
        lw=1.4,
        zorder=2,
    ),
    "scalars_cells": dict(
        label="S1 — scalars_cells",
        color="#adb5bd",   # grey
        ls="--",
        lw=1.2,
        zorder=1,
    ),
}

# ── mapping from W&B run-name prefix to canonical mode key ───────────────────
# Adjust if your W&B run names differ.
PREFIX_TO_MODE = {
    "img_mc_cells_substrates": "img_mc_cells_substrates",
    "img_mc_cells":            "img_mc_cells",
    "spatial_scalars_cells_spatial_substrates": "spatial_scalars_cells_spatial_substrates",
    "spatial_scalars_cells_substrates": "spatial_scalars_cells",  # NOTE: map to S3 key
    "spatial_scalars_cells":   "spatial_scalars_cells",
    "scalars_cells_substrates": "scalars_cells_substrates",
    "scalars_cells":           "scalars_cells",
}

# ── W&B column names (edit these to match your downloaded CSV headers) ────────
WANDB_COLS = {
    "step":        "_step",
    "train_mean50": "charts/train_return_mean50",
    "test_mean50":  "charts/test_return_mean50",
    "rect":         "charts/test_return_rectangle",
    "circ":         "charts/test_return_circular",
}

# ── static per-seed data from report ─────────────────────────────────────────
# Format: {mode_key: [(seed, train50, test50, rect, circ), ...]}
STATIC_DATA = {
    "img_mc_cells_substrates": [
        (1,   78.0,  58.2,  4.4,   106.7),
        (32,  79.7,  69.9,  4.9,   61.4),
        (64,  86.9,  47.7,  109.6, 33.7),
        (64,  64.3,  42.3,  47.3,  128.9),
        (128, 77.1,  64.5,  56.1,  158.8),
    ],
    "img_mc_cells": [
        (1,   39.8,  22.4, -28.7, -28.1),
        (32,  42.4,  27.8, -17.4, -43.9),
        (64,  51.3,  14.5, -0.5,  -2.8),
        (128, 67.5,  18.3, 44.2,  94.8),
    ],
    "spatial_scalars_cells": [
        (1,   16.7, -28.9, 32.2,  -81.3),
        (64,  59.8,  27.3, 223.1,  91.1),
        (128, 77.1,  38.8, 82.9,   39.2),
    ],
    "spatial_scalars_cells_spatial_substrates": [
        (1,   10.4,  1.2,  6.0,  -67.5),
        (64,  -4.1, -0.1, 164.2, -7.6),
    ],
    "scalars_cells_substrates": [
        (1,    1.7,  26.8, 144.6,  -0.6),
        (32,  -79.2, -24.6, -4.8, -90.8),
        (64,  25.3,  18.8,  8.0,  -44.4),
        (128, -15.1, 28.1,  12.9, -54.8),
    ],
    "scalars_cells": [
        (1,   37.2, 24.0, -1.5,  -21.8),
        (32,   6.2, 23.3, 30.3,   46.7),
        (64,  -20.3, 6.9, -52.6, -33.1),
        (128,  0.4, 16.3, -6.9,  -74.0),
    ],
}

MODE_ORDER = [
    "img_mc_cells_substrates",
    "img_mc_cells",
    "spatial_scalars_cells",
    "spatial_scalars_cells_spatial_substrates",
    "scalars_cells_substrates",
    "scalars_cells",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_style():
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  7.5,
        "legend.framealpha": 0.85,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linewidth":   0.5,
    })


def _save(name: str):
    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"{name}.{ext}"
        plt.savefig(out)
    print(f"  saved → {OUTPUT_DIR}/{name}.[pdf|png]")
    plt.close()


def _smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    return uniform_filter1d(arr.astype(float), size=window, mode="nearest")


# ─────────────────────────────────────────────────────────────────────────────
# Static (no CSV) figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_static_bar(metric_idx: int, ylabel: str, title: str, fname: str):
    """Bar chart from STATIC_DATA for a given column index (0=train50 etc.)."""
    _fig_style()
    fig, ax = plt.subplots(figsize=(7, 3.8))

    labels, means, stds, colors = [], [], [], []
    for mode in MODE_ORDER:
        rows = np.array(STATIC_DATA[mode])      # (n_seeds, 5)
        vals = rows[:, metric_idx + 1]           # +1 because col0 = seed
        meta = MODE_META[mode]
        labels.append(meta["label"].split(" — ")[0])   # short label "I2", "S3" etc.
        means.append(vals.mean())
        stds.append(vals.std())
        colors.append(meta["color"])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
                elinewidth=1.2, capsize=4, zorder=4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    legend_handles = [
        mpatches.Patch(color=MODE_META[m]["color"], label=MODE_META[m]["label"])
        for m in MODE_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=1,
              fontsize=6.5, framealpha=0.9)
    _save(fname)


def plot_static_grouped(fname: str = "summary_static"):
    """2-panel bar chart: train50 and test50 side by side."""
    _fig_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    for ax, (col_idx, label, title) in zip(
        axes,
        [
            (0, "Mean training return (last 50 eps)", "Training return"),
            (1, "Mean test return (last 50 eps)",    "Test return"),
        ],
    ):
        labels, means, stds, colors = [], [], [], []
        for mode in MODE_ORDER:
            rows = np.array(STATIC_DATA[mode])
            vals = rows[:, col_idx + 1]
            meta = MODE_META[mode]
            labels.append(meta["label"].split(" — ")[0])
            means.append(vals.mean())
            stds.append(vals.std())
            colors.append(meta["color"])

        x = np.arange(len(labels))
        ax.bar(x, means, color=colors, alpha=0.85, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=4, zorder=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(title)

    # shared legend below figure
    handles = [
        mpatches.Patch(color=MODE_META[m]["color"],
                       label=MODE_META[m]["label"])
        for m in MODE_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    _save(fname)


def plot_layout_comparison(fname: str = "test_layout_comparison"):
    """
    Grouped bar chart: test return by layout (rect, circ, avg).

    Each observation mode gets three adjacent bars — one per layout.
    The three layouts are encoded by BOTH distinct colors AND hatches so
    they remain distinguishable in greyscale print and in the legend.

    Layout colors (fixed, independent of mode):
      Rectangle  — steel blue   #4878d0  hatch //
      Circular   — burnt orange #ee854a  hatch \\\\
      Test avg   — medium green #6acc65  hatch (none, solid)
    """
    # Fixed colors per layout — distinct and colorblind-friendly
    LAYOUT_STYLE = {
        "Rectangle": dict(color="#4878d0", hatch="//",  alpha=0.85),
        "Circular":  dict(color="#ee854a", hatch="\\\\", alpha=0.85),
        "Test avg":  dict(color="#6acc65", hatch="",    alpha=0.85),
    }
    col_idxs = {"Rectangle": 2, "Circular": 3, "Test avg": 1}

    _fig_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))

    n_modes = len(MODE_ORDER)
    bar_w   = 0.26
    x       = np.arange(n_modes)
    offsets = {"Rectangle": -bar_w, "Circular": 0.0, "Test avg": bar_w}

    for layout, style in LAYOUT_STYLE.items():
        col_idx = col_idxs[layout]
        means = []
        for mode in MODE_ORDER:
            rows = np.array(STATIC_DATA[mode])
            means.append(rows[:, col_idx + 1].mean())

        ax.bar(
            x + offsets[layout], means, bar_w,
            color=style["color"],
            hatch=style["hatch"],
            alpha=style["alpha"],
            label=layout,
            zorder=3,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODE_META[m]["label"].split(" — ")[0] for m in MODE_ORDER],
        fontsize=9,
    )
    ax.set_ylabel("Mean return (seeds averaged)", fontsize=9)
    ax.set_title("Test return by held-out layout", fontsize=10)

    # Build legend with explicit patches so color + hatch both show
    legend_handles = [
        mpatches.Patch(
            facecolor=style["color"], hatch=style["hatch"],
            edgecolor="white", label=layout, alpha=style["alpha"],
        )
        for layout, style in LAYOUT_STYLE.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, title="Layout", title_fontsize=9,
              framealpha=0.95)
    fig.tight_layout()
    _save(fname)


# ─────────────────────────────────────────────────────────────────────────────
# W&B CSV figures (learning curves)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_mode(run_name: str) -> str | None:
    """Return mode key from a W&B run name like 'TME_V2_32_img_mc_cells_1234'."""
    lower = run_name.lower()
    # longest-match first
    for prefix in sorted(PREFIX_TO_MODE.keys(), key=len, reverse=True):
        if prefix.lower() in lower:
            return PREFIX_TO_MODE[prefix]
    return None


def load_wandb_histories(wandb_dir: str) -> dict[str, list[pd.DataFrame]]:
    """
    Returns {mode_key: [df_seed1, df_seed2, ...]}
    Each df has columns: step, train_mean50, test_mean50
    """
    wandb_path = Path(wandb_dir)
    mode_dfs: dict[str, list[pd.DataFrame]] = {m: [] for m in MODE_ORDER}

    for run_dir in sorted(wandb_path.iterdir()):
        if not run_dir.is_dir():
            continue
        hist_csv = run_dir / "history.csv"
        if not hist_csv.exists():
            continue

        mode = _infer_mode(run_dir.name)
        if mode is None or mode not in mode_dfs:
            continue

        df = pd.read_csv(hist_csv)

        # rename columns
        col_map = {}
        for key, wname in WANDB_COLS.items():
            if wname in df.columns:
                col_map[wname] = key
        df = df.rename(columns=col_map)

        # keep only needed columns
        keep = [c for c in ["step", "train_mean50", "test_mean50"] if c in df.columns]
        if "step" not in keep:
            continue
        df = df[keep].dropna(subset=["step"])
        df = df.sort_values("step").reset_index(drop=True)
        mode_dfs[mode].append(df)

    return mode_dfs


def _plot_curves(mode_dfs, metric: str, ylabel: str, title: str,
                 ax: plt.Axes, smooth: int = 50):
    """Plot mean ± std curves for one metric on ax."""
    for mode in MODE_ORDER:
        dfs = mode_dfs.get(mode, [])
        if not dfs:
            continue
        if metric not in dfs[0].columns:
            continue

        meta  = MODE_META[mode]
        steps_list = [df["step"].values for df in dfs]
        # common step grid: union of all steps
        all_steps = np.unique(np.concatenate(steps_list))

        interp_vals = []
        for df in dfs:
            y = df[metric].values.astype(float)
            y = np.interp(all_steps, df["step"].values, y)
            interp_vals.append(_smooth(y, smooth))

        arr   = np.array(interp_vals)     # (n_seeds, T)
        mean  = arr.mean(axis=0)
        std   = arr.std(axis=0)

        ax.plot(all_steps, mean,
                color=meta["color"], ls=meta["ls"], lw=meta["lw"],
                label=meta["label"], zorder=meta["zorder"])
        ax.fill_between(all_steps, mean - std, mean + std,
                        color=meta["color"], alpha=0.15, zorder=meta["zorder"] - 1)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=0)


def plot_wandb_curves(wandb_dir: str):
    """Full learning-curve figure for the paper."""
    _fig_style()
    mode_dfs = load_wandb_histories(wandb_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    _plot_curves(mode_dfs, "train_mean50",
                 "Training return (mean50)", "Training return",
                 axes[0])
    _plot_curves(mode_dfs, "test_mean50",
                 "Test return (mean50)", "Test return (held-out layouts)",
                 axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   fontsize=7, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save("train_return_mean50")

    # save individual panels too
    for ax, fname in zip(axes, ["train_return_mean50", "test_return_mean50"]):
        figsi, axsi = plt.subplots(figsize=(5.5, 4.0))
        _plot_curves(mode_dfs, fname.replace("train_return_mean50", "train_mean50")
                                    .replace("test_return_mean50", "test_mean50"),
                     ax.get_ylabel(), ax.get_title(), axsi)
        axsi.legend(fontsize=7, loc="upper left")
        _save(fname)
        plt.close(figsi)

    plt.close(fig)


def plot_wandb_test_layouts(wandb_dir: str):
    """Separate curves for rectangle and circular test layouts."""
    _fig_style()
    mode_dfs = load_wandb_histories(wandb_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    _plot_curves(mode_dfs, "rect",
                 "Return", "Test return — Rectangle layout", axes[0])
    _plot_curves(mode_dfs, "circ",
                 "Return", "Test return — Circular layout", axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   fontsize=7, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save("test_layouts_curves")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate EWRL 2026 paper figures")
    parser.add_argument("--source", choices=["wandb", "static"], default="static",
                        help="'wandb' to read downloaded CSV histories, "
                             "'static' to use hard-coded per-seed table")
    parser.add_argument("--wandb_dir", default="wandb_csv_exports",
                        help="Directory with downloaded W&B history CSVs "
                             "(used only when --source wandb)")
    args = parser.parse_args()

    print(f"Generating figures → {OUTPUT_DIR}")

    if args.source == "static":
        print("Mode: static (no CSV required)")
        plot_static_grouped("summary_static")
        plot_layout_comparison("test_layout_comparison")
        # Also generate individual bar charts
        plot_static_bar(0, "Mean training return (last 50 eps)",
                        "Training return by observation mode", "train_return_mean50")
        plot_static_bar(1, "Mean test return (last 50 eps)",
                        "Test return by observation mode", "test_return_mean50")
        print("Done. To get proper learning curves, run with --source wandb "
              "after downloading W&B histories.")
    else:
        print(f"Mode: W&B CSV exports from {args.wandb_dir}")
        plot_wandb_curves(args.wandb_dir)
        plot_wandb_test_layouts(args.wandb_dir)
        # Also generate static summary for supplementary
        plot_static_grouped("summary_static")
        plot_layout_comparison("test_layout_comparison")


if __name__ == "__main__":
    main()
