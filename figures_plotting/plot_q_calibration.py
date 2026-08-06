"""Value-calibration figure for Ch. 6 §6.3, both transfer directions.

Two panels, one per direction, sharing a y-axis: mean over 5 seeds of the
smoothed signed calibration error test_q_bias = mean(Q - MC) on held-out test
episodes, against training step.

Why a trajectory and not a bar chart: the signed bias is strongly non-monotone.
Scalar critics overestimate held-out value by a wide margin in mid-training and
are then dragged back down -- SAC evaluates min(qf1, qf2), which is deliberately
pessimistic, so end-of-training bias understates the miscalibration that
occurred.  A single end-of-training number hides the entire effect.

Style follows plot_tme_new.py (palette, EWMA, mean +- 1 sd shading).
"""

import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyse_q_calibration import _ewma, _seed_id, MODE_ID

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out_q_calibration")
os.makedirs(OUT, exist_ok=True)

# wandb's _step is the drained-transition counter (verified: _step == samples_drained
# exactly), i.e. agent decisions, which top out at 1e5.  Each decision is held for
# ACTION_REPEAT simulator steps, giving the 6e5 simulator-step budget Ch. 6 quotes.
# Plot in simulator steps so this figure shares an x-axis with plot_tme_new.py.
ACTION_REPEAT = 6

PANELS = [
    ("wandb_train_rectangle_test_networkfield",
     "train rectangle $\\rightarrow$ test network-field"),
    ("wandb_train_networkfield_test_rectangle",
     "train network-field $\\rightarrow$ test rectangle"),
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


def seeds(direction, mode, col="q_bias"):
    out = []
    files = sorted(glob.glob(os.path.join(BASE, direction, mode, "SAC_*.csv")),
                   key=_seed_id)[:5]
    for f in files:
        df = pd.read_csv(f, usecols=["step", col]).dropna().sort_values("step")
        if len(df) > 10:
            out.append((df["step"].to_numpy(), _ewma(df[col].to_numpy(), 20)))
    return out


def main():
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 120, "savefig.dpi": 300,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)

    for ax, (direction, title) in zip(axes, PANELS):
        for mode, (mid, colour, ls, lw) in STYLE.items():
            ser = seeds(direction, mode)
            if not ser:
                continue
            hi = min(s[-1] for s, _ in ser)
            grid = np.linspace(0, hi, 300)
            mat = np.array([np.interp(grid, s, v) for s, v in ser])
            mu, sd = mat.mean(0), mat.std(0)
            x = grid * ACTION_REPEAT / 1e5
            ax.plot(x, mu, color=colour, ls=ls, lw=lw, label=mid, zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=colour, alpha=0.10,
                            lw=0, zorder=1)
        ax.axhline(0, color="k", lw=0.8, alpha=0.6, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"cumulative simulator steps ($\times 10^5$)")

    axes[0].set_ylabel(r"$\mathbb{E}[\,Q - G_t^{\mathrm{MC}}\,]$  (test episodes)")
    axes[0].annotate("critic overestimates held-out value", xy=(0.03, 0.94),
                     xycoords="axes fraction", fontsize=8.5, alpha=0.75)
    axes[0].annotate("underestimates", xy=(0.03, 0.03), xycoords="axes fraction",
                     fontsize=8.5, alpha=0.75)
    axes[1].legend(ncol=1, fontsize=8, loc="center left",
                   bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"q_bias_both_directions.{ext}"))
    print(f"wrote {OUT}/q_bias_both_directions.pdf")


if __name__ == "__main__":
    main()
