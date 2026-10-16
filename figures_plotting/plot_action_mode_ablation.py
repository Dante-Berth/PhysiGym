"""Action-space ablation: targeted vs. full (uniform) dosing, both directions.

The figure Ch. 6 §6.1 is a stub for.  Q1 asks whether the *targeted* action
space -- dosing only near the tumour -- is what buys the performance, or whether
a policy over the full uniform action space does just as well.  Answering it
needs no checkpoint: both sweeps logged returns, and returns are the whole
question.

    targeted   SAC_ASYNC_TME_NEW_HYP_REWARD{,_TRAIN_NETWORKFIELD_TEST_RECTANGLE}
    full       SAC_ASYNC_TME_NEW_HYP_REWARD_FULL_OLDMODEL
               ..._TRAIN_NETWORKFIELD_TEST_RECTANGLE_FULL_OLDMODEL
               (2026-10, pre-chemotaxis model, S1/S3m/I2 only; replaces the
               new-model sweeps wandb_full_*, which are kept on disk)

⚠️ The `full` projects are NOT returned by ``wandb.Api().projects()`` -- see
RESEARCH_LOG T1 and download_wandb_project_full.py.  They were presumed not to
exist for exactly that reason.  They do; 87 runs.

AGGREGATION IS IMPORTED, NOT REIMPLEMENTED.  Every statistical choice --
EWMA-50 per seed, five distinct seeds by lowest seed-id, 95% bootstrap-of-the-
mean over B=10,000 seed resamples, columns kept only where every seed has data
-- comes from plot_learning_curves_both_directions by import.  If that file's
convention changes, this figure changes with it; they cannot disagree.

Encoding: HUE = observation mode (inherited unchanged from the targeted figure,
so a mode is the same colour in both).  LINESTYLE = action space -- solid for
targeted, dashed for full.  That is the one comparison the figure exists to
make, so it gets the channel that survives greyscale printing.

Outputs (out_action_mode_ablation/):
    action_mode_ablation.pdf/.png   2x2, rows = direction, cols = train/test
    action_mode_collapse.pdf/.png   the diagnostic: per-mode final return and
                                    mean action step, targeted vs full.  Shows
                                    the collapse directly instead of leaving it
                                    to be inferred from overlapping curves.
    action_mode_endpoints.csv       per mode/split/direction, both action modes
                                    and their difference -- the numbers a caption
                                    or §6.1 sentence may quote

Run from figures_plotting/.
"""
import glob
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_learning_curves_both_directions as base
from plot_learning_curves_both_directions import (
    BASE, ACTION_REPEAT, MAX_SEEDS, STEP_MAX, NGRID,
    MODE_META, MODE_ORDER, _ewma, _seed_id, bootstrap_band, _style,
)

OUT = os.path.join(BASE, "out_action_mode_ablation")
os.makedirs(OUT, exist_ok=True)

# dir + restrict per (direction, action_mode).  `restrict` pins the targeted
# rect2net sweep to the run-ids the reported tables came from, exactly as the
# targeted figure does; the full sweeps are whole projects and need no pinning.
SWEEPS = {
    ("rect2net", "targeted"): dict(
        dir="wandb_train_rectangle_test_networkfield",
        restrict="wandb_tme_new/manifest.csv"),
    ("rect2net", "full"): dict(
        dir="wandb_oldmodel_full_train_rectangle_test_networkfield", restrict=None),
    ("net2rect", "targeted"): dict(
        dir="wandb_train_networkfield_test_rectangle", restrict=None),
    ("net2rect", "full"): dict(
        dir="wandb_oldmodel_full_train_networkfield_test_rectangle", restrict=None),
}

DIRECTION_META = {
    "rect2net": dict(row=r"rectangle $\rightarrow$ network-field",
                     train="Train (rectangle)",
                     test="Test (network-field, held out)"),
    "net2rect": dict(row=r"network-field $\rightarrow$ rectangle",
                     train="Train (network-field)",
                     test="Test (rectangle, held out)"),
}

ACTION_STYLE = {
    "targeted": dict(ls="-",  lw=1.7, alpha=1.00),
    "full":     dict(ls="--", lw=1.5, alpha=0.95),
}

# A run that died in the first few hundred steps is not a seed, it is a crash.
# The net2rect full sweep has four such stubs in img_mc_cells_substrates
# (38-105 rows against ~3,400 for a real run); taking them as the lowest
# seed-ids would silently replace good seeds with noise.
MIN_ROWS = 500


def seed_files(direction, action_mode, mode):
    """Run CSVs for one cell: <=5 distinct seeds, lowest seed-id, crashes dropped."""
    cfg = SWEEPS[(direction, action_mode)]
    d = os.path.join(BASE, cfg["dir"], mode)
    prefix = "RANDOM_" if mode == "random_baseline" else "SAC_"
    files = sorted(glob.glob(os.path.join(d, prefix + "*.csv")), key=_seed_id)
    if cfg["restrict"]:
        keep = set(pd.read_csv(os.path.join(BASE, cfg["restrict"])).run_id)
        files = [f for f in files
                 if os.path.basename(f).rsplit("_", 1)[-1][:-4] in keep]
    files = [f for f in files if sum(1 for _ in open(f)) - 1 >= MIN_ROWS]
    seen, dedup = set(), []
    for f in files:                      # deterministic: first run-id per seed
        s = _seed_id(f)
        if s not in seen:
            seen.add(s)
            dedup.append(f)
    return dedup[:MAX_SEEDS]


def seed_matrix(direction, action_mode, mode, col, grid):
    rows = []
    for f in seed_files(direction, action_mode, mode):
        d = pd.read_csv(f, usecols=["step", col]).dropna().sort_values("step")
        if len(d) <= 3:
            continue
        rows.append(np.interp(grid, d["step"].to_numpy(), _ewma(d[col].to_numpy()),
                              left=np.nan, right=np.nan))
    return np.array(rows) if rows else np.empty((0, len(grid)))


ABLATION_MODES = ["scalars_macrophages",
                  "spatial_scalars_cells_m1m2", "img_mc_cells_substrates"]


def draw(fname="action_mode_ablation"):
    """2 x 4: rows = direction; columns = targeted (train, test), uniform (train, test)."""
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), sharex=True, sharey="row")
    YLIM = {"rect2net": (-165, 75), "net2rect": (-85, 85)}
    grid = np.linspace(0, STEP_MAX, NGRID)
    x = grid * ACTION_REPEAT / 1e5

    for r, direction in enumerate(DIRECTION_META):
        cfg = DIRECTION_META[direction]
        for ci, (action_mode, col) in enumerate(
                [(a, c) for a in ("targeted", "full")
                 for c in ("train_return", "test_return")]):
            ax = axes[r, ci]
            st = ACTION_STYLE[action_mode]
            for mode in ABLATION_MODES:
                mat = seed_matrix(direction, action_mode, mode, col, grid)
                if mat.size == 0:
                    continue
                m = MODE_META[mode]
                mean, lo, hi, _ = bootstrap_band(mat)
                ax.plot(x, mean, color=m["color"], ls="-", lw=1.7,
                        zorder=m["z"], label=m["id"])
                ax.fill_between(x, lo, hi, color=m["color"], alpha=0.12,
                                zorder=m["z"] - 0.5, linewidth=0)
            ax.axhline(0, color="k", lw=0.6, alpha=0.35, zorder=0)
            ax.set_ylim(*YLIM[direction])
            name = "uniform" if action_mode == "full" else "targeted"
            split = cfg["train"] if col == "train_return" else cfg["test"]
            ax.set_title(f"{name}: {split}", pad=4, fontsize=9.5)
            if ci == 0:
                ax.set_ylabel("cumulative return")
                ax.text(-0.30, 0.5, cfg["row"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10.5,
                        fontweight="bold")
            if r == 1:
                ax.set_xlabel(r"simulator steps ($\times 10^{5}$)")

    mode_h = [Line2D([], [], color=MODE_META[m]["color"], ls="-", lw=1.7,
                     label=MODE_META[m]["id"]) for m in ABLATION_MODES]
    fig.legend(handles=mode_h, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=9.5)
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"))
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png")


def endpoints():
    """Table convention (last 50 train / 20 test smoothed points), both modes.

    Same statistic as the mu columns of Tables 5.1 / 5.3, so `targeted` here
    should reproduce those numbers and `full` is directly comparable to them.
    """
    rows = []
    for direction in DIRECTION_META:
        for col, tail in [("train_return", 50), ("test_return", 20)]:
            for mode in MODE_ORDER:
                rec = dict(direction=direction, split=col.split("_")[0],
                           id=MODE_META[mode]["id"])
                for action_mode in ("targeted", "full"):
                    vals = []
                    for f in seed_files(direction, action_mode, mode):
                        d = pd.read_csv(f, usecols=["step", col]).dropna().sort_values("step")
                        if len(d) > 3:
                            vals.append(_ewma(d[col].to_numpy())[-tail:].mean())
                    rec[f"{action_mode}_n"] = len(vals)
                    rec[f"{action_mode}_mu"] = np.mean(vals) if vals else np.nan
                rec["delta_full_minus_targeted"] = (
                    rec["full_mu"] - rec["targeted_mu"])
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "action_mode_endpoints.csv"), index=False)
    return df


def collapse(fname="action_mode_collapse"):
    """Slope chart: final training return, targeted -> uniform, one line per mode.

    One panel per transfer direction.  Each mode is a line from its targeted
    endpoint to its uniform endpoint, mean over seeds with a 95% bootstrap CI
    (B = 10,000 seed resamples) and the individual seeds as small dots.  Identity
    is carried by colour, marker shape and a direct label, never colour alone.
    Replaces the 2026-10-16 dot plot, which put two action spaces and an
    action-step panel on one axis and was hard to read.
    """
    _style()
    rng = np.random.default_rng(0)
    modes = [m for m in ABLATION_MODES if m != "random_baseline"]
    label = {"scalars_macrophages": "S1 (blind)",
             "spatial_scalars_cells_m1m2": "S3m (spatial scalars)",
             "img_mc_cells_substrates": "I2 (image)"}
    marker = {"scalars_macrophages": "o", "spatial_scalars_cells_m1m2": "s",
              "img_mc_cells_substrates": "D"}
    ink = {"scalars_macrophages": "#5c677d", "spatial_scalars_cells_m1m2": "#0077b6",
           "img_mc_cells_substrates": "#c1121f"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    for ax, direction in zip(axes, DIRECTION_META):
        ends = {}
        for mode in modes:
            mu, lo, hi, pts = [], [], [], []
            for action_mode in ("targeted", "full"):
                v = []
                for f in seed_files(direction, action_mode, mode):
                    d = pd.read_csv(f, usecols=["step", "train_return"]).dropna().sort_values("step")
                    if len(d) > 3:
                        v.append(_ewma(d["train_return"].to_numpy())[-50:].mean())
                v = np.array(v)
                boot = rng.choice(v, size=(10_000, len(v)), replace=True).mean(1)
                mu.append(v.mean()); lo.append(np.percentile(boot, 2.5))
                hi.append(np.percentile(boot, 97.5)); pts.append(v)
            c = ink[mode]
            xs = np.array([0.0, 1.0])
            for xi, v in zip(xs, pts):
                jit = (rng.random(len(v)) - 0.5) * 0.10
                ax.scatter(xi + jit, v, s=12, color=c, alpha=0.30, lw=0, zorder=2)
            ax.plot(xs, mu, color=c, lw=2.0, zorder=3)
            ax.errorbar(xs, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], fmt=marker[mode],
                        ms=8, color=c, mec="white", mew=1.2, elinewidth=1.4, capsize=3, zorder=4)
            ends[mode] = mu[1]
        # direct labels at the uniform end, nudged apart when they collide
        order = sorted(modes, key=lambda m: ends[m])
        ys, last = {}, -1e9
        for m in order:
            y = max(ends[m], last + 9.0); ys[m] = y; last = y
        for m in modes:
            ax.text(1.12, ys[m], label[m], color=ink[m], va="center", fontsize=9)
        ax.axhline(0, color="k", lw=0.6, alpha=0.35, zorder=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["targeted\naction", "uniform\naction"])
        ax.set_xlim(-0.25, 1.75)
        ax.set_title(DIRECTION_META[direction]["row"], fontsize=10.5, pad=6)
        ax.grid(axis="x", visible=False); ax.grid(axis="y", alpha=0.22)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("final training return")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"))
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png")


if __name__ == "__main__":
    draw()
    collapse()
    df = endpoints()
    print("\nTable-convention means, targeted vs. full:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print("\nOutputs in", OUT)
