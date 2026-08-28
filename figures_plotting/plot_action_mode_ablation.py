"""Action-space ablation: targeted vs. full (uniform) dosing, both directions.

The figure Ch. 6 §6.1 is a stub for.  Q1 asks whether the *targeted* action
space -- dosing only near the tumour -- is what buys the performance, or whether
a policy over the full uniform action space does just as well.  Answering it
needs no checkpoint: both sweeps logged returns, and returns are the whole
question.

    targeted   SAC_ASYNC_TME_NEW_HYP_REWARD{,_TRAIN_NETWORKFIELD_TEST_RECTANGLE}
    full       ..._TRAIN_RECTANGLE_TEST_NETWORKFIELD_NEW_CHEMO
               ..._TRAIN_NETWORK_FIELD_TEST_RECTANGLE_full

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
        dir="wandb_full_train_rectangle_test_networkfield", restrict=None),
    ("net2rect", "targeted"): dict(
        dir="wandb_train_networkfield_test_rectangle", restrict=None),
    ("net2rect", "full"): dict(
        dir="wandb_full_train_networkfield_test_rectangle", restrict=None),
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


def draw(fname="action_mode_ablation"):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True, sharey="all")
    grid = np.linspace(0, STEP_MAX, NGRID)
    x = grid * ACTION_REPEAT / 1e5

    for r, direction in enumerate(DIRECTION_META):
        cfg = DIRECTION_META[direction]
        for c, col in enumerate(["train_return", "test_return"]):
            ax = axes[r, c]
            for action_mode in ("targeted", "full"):
                st = ACTION_STYLE[action_mode]
                for mode in MODE_ORDER:
                    mat = seed_matrix(direction, action_mode, mode, col, grid)
                    if mat.size == 0:
                        continue
                    m = MODE_META[mode]
                    mean, lo, hi, _ = bootstrap_band(mat)
                    ax.plot(x, mean, color=m["color"], ls=st["ls"], lw=st["lw"],
                            alpha=st["alpha"], zorder=m["z"], label=m["id"])
                    ax.fill_between(x, lo, hi, color=m["color"], alpha=0.10,
                                    zorder=m["z"] - 0.5, linewidth=0)
            ax.axhline(0, color="k", lw=0.6, alpha=0.35, zorder=0)
            ax.set_title(cfg["train"] if c == 0 else cfg["test"], pad=4)
            if c == 0:
                ax.set_ylabel("cumulative return")
                ax.text(-0.20, 0.5, cfg["row"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10.5,
                        fontweight="bold")
            if r == 1:
                ax.set_xlabel(r"cumulative simulator steps ($\times 10^{5}$)")

    mode_h = [Line2D([], [], color=MODE_META[m]["color"], ls="-", lw=1.7,
                     label=MODE_META[m]["id"]) for m in MODE_ORDER]
    act_h = [Line2D([], [], color="0.25", ls=ACTION_STYLE[a]["ls"],
                    lw=ACTION_STYLE[a]["lw"], label=f"{a} action space")
             for a in ("targeted", "full")]
    fig.legend(handles=mode_h + act_h, loc="lower center", ncol=12, frameon=False,
               bbox_to_anchor=(0.5, -0.045), fontsize=9.0, columnspacing=1.3,
               handlelength=2.4)
    fig.tight_layout(rect=(0.015, 0.025, 1, 1))
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
    """Per-mode dot plot: final return, and mean per-step action change.

    The learning-curve figure shows the full-action curves lying on top of one
    another, which is easy to mistake for overplotting.  This shows the spread
    as a number per mode.  The right panel is the mechanism: `action_delta_mean`
    is the mean absolute change in the action between consecutive decisions, so
    a policy that has stopped responding to its observation reports a small,
    mode-independent value.

    ⚠️ The two action spaces have different dimension (targeted 4-D, full 1-D),
    so the delta panel is NOT a like-for-like magnitude comparison and no claim
    is made about which number is "larger".  What is comparable, and what the
    panel is for, is the SPREAD ACROSS MODES within one action space.
    """
    _style()
    modes = [m for m in MODE_ORDER if m != "random_baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    y = np.arange(len(modes))[::-1]

    for ax, (col, tail, lab) in zip(axes, [
            ("train_return", 50, "final training return"),
            ("train_action_delta_mean", 50, r"mean action step $|\Delta a|$")]):
        for action_mode, mk in (("targeted", "o"), ("full", "s")):
            xs = []
            for mode in modes:
                v = []
                for f in seed_files("rect2net", action_mode, mode):
                    d = pd.read_csv(f, usecols=["step", col]).dropna().sort_values("step")
                    if len(d) > 3:
                        v.append(_ewma(d[col].to_numpy())[-tail:].mean())
                xs.append(np.mean(v) if v else np.nan)
            xs = np.array(xs)
            ax.plot(xs, y, mk, ms=6, ls="none",
                    mfc="none" if action_mode == "full" else None,
                    color="#1d3557" if action_mode == "targeted" else "#e63946",
                    label=f"{action_mode}")
            ok = ~np.isnan(xs)
            if ok.sum() > 1:
                ax.annotate("", xy=(xs[ok].min(), -0.72), xytext=(xs[ok].max(), -0.72),
                            arrowprops=dict(arrowstyle="<->", lw=1.0,
                                            color="#1d3557" if action_mode == "targeted" else "#e63946"))
                ax.text((xs[ok].min() + xs[ok].max()) / 2, -1.15,
                        f"spread {xs[ok].max() - xs[ok].min():.3g}", ha="center",
                        fontsize=8.5,
                        color="#1d3557" if action_mode == "targeted" else "#e63946")
        ax.set_yticks(y)
        ax.set_yticklabels([MODE_META[m]["id"] for m in modes])
        ax.set_ylim(-1.9, len(modes) - 0.4)
        ax.set_xlabel(lab)
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle("rectangle $\\rightarrow$ network-field: modes separate under targeted "
                 "action, collapse under uniform", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
