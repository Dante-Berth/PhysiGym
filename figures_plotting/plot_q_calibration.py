"""Value-calibration figure for Ch. 6 §6.3, both transfer directions.

Row 1: signed calibration error test_q_bias = mean(Q - MC) on held-out test
episodes, mean over 5 seeds +- 1 sd, against cumulative simulator steps.

Row 2 (omit with --bias-only): the critic's own TD objective qf1_loss on the *replay
buffer*, i.e. on-distribution.  It is the control for row 1: a critic that is
"confidently wrong out of distribution" must fit its own training targets, and
these do not.  Log scale, because the families differ by two orders of
magnitude.

Why a trajectory and not a bar chart: the signed bias is strongly non-monotone.
Scalar critics overestimate held-out value by a wide margin in mid-training and
are then dragged back down -- SAC evaluates min(qf1, qf2), which is deliberately
pessimistic, so end-of-training bias understates the miscalibration that
occurred.  A single end-of-training number hides the entire effect.

Two things this script got wrong before 2026-08-21, both fixed here:

  * It re-derived its own run selection as sorted(...)[:5], skipping both the
    manifest restriction and the seed de-duplication that analyse_q_calibration
    applies.  Seven of the nine rect->net modes were then drawn from a different
    set of runs than the tables quote, I2 from five relaunches of seed 1 alone,
    and the crashed relaunches it admitted truncated the drawn x-range to as
    little as 5% of the axis (S3m).  It now calls select_runs().
  * It built its interpolation grid from 0, but no run logs q_bias before ~2650
    agent decisions.  np.interp clamps, so everything left of a seed's first
    logged point was a horizontal line at that seed's first value -- fabricated,
    and fabricated from the noisiest point in the run.  The grid now starts at
    the last seed's first logged step.

Style follows plot_tme_new.py (palette, EWMA, mean +- 1 sd shading).
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyse_q_calibration import _ewma, select_runs

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out_q_calibration")
os.makedirs(OUT, exist_ok=True)

# wandb's _step is the drained-transition counter (verified: _step == samples_drained
# exactly), i.e. agent decisions, which top out at 1e5.  Each decision is held for
# ACTION_REPEAT simulator steps, giving the 6e5 simulator-step budget Ch. 6 quotes.
# Plot in simulator steps so this figure shares an x-axis with plot_tme_new.py.
ACTION_REPEAT = 6

PANELS = [
    ("rect2net", "train rectangle $\\rightarrow$ test network-field"),
    ("net2rect", "train network-field $\\rightarrow$ test rectangle"),
]

STYLE = {
    "img_mc_cells_substrates_m1m2": ("I2m", "#b5179e", "-", 1.9),
    "img_mc_cells_substrates":      ("I2",  "#e63946", "-", 1.9),
    "img_mc_cells_m1m2":            ("I1m", "#f3722c", "-", 1.5),
    "img_mc_cells":                 ("I1",  "#2a9d8f", "-", 1.7),
    "spatial_scalars_cells_substrates_m1m2": ("S3sm", "#3a0ca3", "--", 1.5),
    "spatial_scalars_cells_substrates":      ("S3s",  "#4361ee", "--", 1.5),
    "spatial_scalars_cells_m1m2":            ("S3m",  "#4895ef", "--", 1.4),
    "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2":
                                             ("S5m",  "#57cc99", ":", 1.5),
    "scalars_macrophages":          ("POMDP", "#adb5bd", "-.", 1.4),
}


def seeds(direction, mode, col):
    """Smoothed (step, value) per selected run.  Selection is select_runs(), never
    a locally re-derived one -- see the module docstring."""
    out = []
    for f in select_runs(direction, mode):
        df = pd.read_csv(f, usecols=lambda c: c in ("step", col))
        if col not in df.columns:
            continue
        df = df[["step", col]].dropna().sort_values("step")
        if len(df) > 10:
            out.append((df["step"].to_numpy(), _ewma(df[col].to_numpy(), 20)))
    return out


def band(ax, direction, col, log=False):
    """Mean +- 1 sd over seeds, on the span where every seed has real data.

    log=True aggregates in log space (geometric mean, multiplicative band).  On a
    log axis an arithmetic mu - sd goes negative whenever sd > mu, which is most
    of the scalar family here; matplotlib then clips the band at the axis floor
    and paints a solid wash over the panel that reads as data.
    """
    for mode, (mid, colour, ls, lw) in STYLE.items():
        ser = seeds(direction, mode, col)
        if not ser:
            continue
        # Interpolate only inside the observed range of every seed.  Starting the
        # grid at 0 would make np.interp clamp to each seed's first logged value
        # and draw a flat segment that is not data.
        lo = max(s[0] for s, _ in ser)
        hi = min(s[-1] for s, _ in ser)
        grid = np.linspace(lo, hi, 300)
        mat = np.array([np.interp(grid, s, v) for s, v in ser])
        if log:
            lg = np.log10(np.clip(mat, 1e-12, None))
            mu = 10 ** lg.mean(0)
            band_lo = 10 ** (lg.mean(0) - lg.std(0))
            band_hi = 10 ** (lg.mean(0) + lg.std(0))
        else:
            mu, sd = mat.mean(0), mat.std(0)
            band_lo, band_hi = mu - sd, mu + sd
        x = grid * ACTION_REPEAT / 1e5
        ax.plot(x, mu, color=colour, ls=ls, lw=lw, label=mid, zorder=3)
        ax.fill_between(x, band_lo, band_hi, color=colour, alpha=0.10, lw=0, zorder=1)


def main(with_loss):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 120, "savefig.dpi": 300,
                         "savefig.bbox": "tight"})
    nrow = 2 if with_loss else 1
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 4.0 * nrow), squeeze=False)

    for j, (direction, title) in enumerate(PANELS):
        ax = axes[0][j]
        band(ax, direction, "q_bias")
        ax.axhline(0, color="k", lw=0.8, alpha=0.6, zorder=2)
        ax.set_title(title, fontsize=10)
        if not with_loss:
            ax.set_xlabel(r"cumulative simulator steps ($\times 10^5$)")
    lo = min(a.get_ylim()[0] for a in axes[0])
    hi = max(a.get_ylim()[1] for a in axes[0])
    for a in axes[0]:
        a.set_ylim(lo, hi)
    axes[0][1].set_yticklabels([])
    axes[0][0].set_ylabel(r"$\mathbb{E}[\,Q - G_t^{\mathrm{MC}}\,]$  (test episodes)")
    axes[0][0].annotate("critic overestimates held-out value", xy=(0.03, 0.94),
                        xycoords="axes fraction", fontsize=8.5, alpha=0.75)
    axes[0][0].annotate("underestimates", xy=(0.03, 0.03),
                        xycoords="axes fraction", fontsize=8.5, alpha=0.75)

    if with_loss:
        for j, (direction, _) in enumerate(PANELS):
            ax = axes[1][j]
            band(ax, direction, "qf1_loss", log=True)
            ax.set_yscale("log")
            ax.set_xlabel(r"cumulative simulator steps ($\times 10^5$)")
        lo = min(a.get_ylim()[0] for a in axes[1])
        hi = max(a.get_ylim()[1] for a in axes[1])
        for a in axes[1]:
            a.set_ylim(lo, hi)
        axes[1][1].set_yticklabels([])
        axes[1][0].set_ylabel(r"critic TD loss $\mathcal{L}(Q_{\phi_1})$  (replay buffer)")

    axes[0][1].legend(ncol=1, fontsize=8, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout()
    stem = "q_bias_both_directions" if with_loss else "q_bias_only_both_directions"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    print(f"wrote {OUT}/{stem}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias-only", action="store_true",
                    help="drop the on-distribution TD-loss row (pre-2026-08-21 layout)")
    a = ap.parse_args()
    main(not a.bias_only)
