#!/usr/bin/env python
"""Forest plots of the paired spatial-axis effects (Ch. 6 §6.5).

THREE figures, not one grid, because they make three different points and a 2x2
grid leaves the reader to work out which is which:

  paired_critic.pdf   the result that holds.  TD residual and calibration, both
                      directions, uniformly one-sided.  28/28 and 26/28.
  paired_return.pdf   the result that does not.  Held-out return and the
                      generalisation gap change sign with transfer direction,
                      which is why the pooled p = 0.09 is not the headline.
  paired_action.pdf   the ablation.  The same paired effect under targeted and
                      uniform action: the TD residual survives, calibration
                      collapses in magnitude.

Each row is a (content pair, transfer direction) cell: the mean paired delta with
a 95% bootstrap CI, the per-seed deltas as light scatter behind it, and a bold
aggregate row carrying sign consistency and p.  Blue runs with the prediction,
red against it.

Reads out_paired_spatial/paired_deltas{,_action}.csv from paired_spatial_tests.py.
Matplotlib only: TikZ is not installed on this machine and packages.tex loads it
conditionally, so a TikZ figure would render as nothing, silently.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paired_spatial_tests import (
    OUT, METRICS, LABEL, PRED, sign_test, boot_ci, PAIRS,
)

DIR_SHORT = {"rect2net": r"rect $\rightarrow$ net",
             "net2rect": r"net $\rightarrow$ rect"}
# Panels: the four metrics that carry a directional prediction.
PANELS = ["test_return", "gap", "td_residual", "q_mae"]
XLABEL = {
    "test_return": r"Held-out return   paired $\Delta$ (image $-$ scalar)",
    "gap": r"Generalisation gap   paired $\Delta$ (image $-$ scalar)",
    "td_residual": r"TD residual $\sqrt{\mathcal{L}(Q_{\phi_1})}$   paired $\Delta\,\log$",
    "q_mae": r"Calibration $m$   paired $\Delta\,\log$",
}
POS, NEG = "#e63946", "#4361ee"     # red = favours scalar, blue = favours image


def draw(ax, sub, spec, xlabel, group_col="pair", extra=None):
    """One forest panel.  Rows are (pair, direction); bold aggregate row last."""
    rows, y = [], 0.0
    for tag, _, _, _ in PAIRS:
        for direction in ("rect2net", "net2rect"):
            cell = sub[(sub.pair == tag) & (sub.direction == direction)]
            if cell.empty:
                continue
            rows.append((y, f"{tag}  {DIR_SHORT[direction]}", cell.delta.to_numpy()))
            y += 1.0
        y += 0.35

    for yy, _lab, d in rows:
        m = d.mean()
        favours = (m < 0) if spec["h1"] == "less" else (m > 0)
        c = NEG if favours else POS
        ax.scatter(d, np.full_like(d, yy), s=14, alpha=0.35, color=c,
                   zorder=2, linewidths=0)
        lo, hi = boot_ci(d)
        ax.plot([lo, hi], [yy, yy], color=c, lw=1.8, zorder=3,
                solid_capstyle="round")
        ax.plot([m], [yy], "o", color=c, ms=6, zorder=4)

    d_all = sub.delta.to_numpy()
    k, n, p = sign_test(d_all, spec["h1"])
    lo, hi = boot_ci(d_all)
    y_agg = y + 0.25
    ax.plot([lo, hi], [y_agg, y_agg], color="k", lw=2.4, zorder=5,
            solid_capstyle="round")
    ax.plot([d_all.mean()], [y_agg], "D", color="k", ms=7, zorder=6)
    rows.append((y_agg, f"All ({k}/{n}, p={p:.1g})", d_all))

    ax.axvline(0, ls="--", lw=1.0, color="0.35", zorder=1)
    ax.set_yticks([r[0] for r in rows])
    labels = [r[1] for r in rows]
    ax.set_yticklabels(labels)
    for tick, lab in zip(ax.get_yticklabels(), labels):
        if lab.startswith("All"):
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(f"Prediction: {PRED[spec['h1']]}" + (f"   {extra}" if extra else ""),
                 fontsize=9, loc="left")
    ax.margins(y=0.06)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    print(f"wrote {OUT}/{name}.pdf")


def fig_pair(df, metrics, name):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, metric in zip(axes, metrics):
        draw(ax, df[df.metric == metric], METRICS[metric], XLABEL[metric])
    save(fig, name)


def fig_action(act):
    """The ablation: same pairs, targeted vs uniform, critic metrics only."""
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.0))
    for row, metric in enumerate(("td_residual", "q_mae")):
        for col, action in enumerate(("targeted", "full")):
            sub = act[(act.metric == metric) & (act.action == action)]
            name = "targeted action" if action == "targeted" else "uniform action"
            draw(axes[row, col], sub, METRICS[metric], XLABEL[metric], extra=name)
    # a shared x-range per row makes the collapse visible as a shift, not a rescale
    for row in range(2):
        lo = min(axes[row, c].get_xlim()[0] for c in range(2))
        hi = max(axes[row, c].get_xlim()[1] for c in range(2))
        for c in range(2):
            axes[row, c].set_xlim(lo, hi)
    save(fig, "paired_action")


def main():
    df = pd.read_csv(os.path.join(OUT, "paired_deltas.csv"))
    fig_pair(df, ["td_residual", "q_mae"], "paired_critic")
    fig_pair(df, ["test_return", "gap"], "paired_return")
    fig_action(pd.read_csv(os.path.join(OUT, "paired_deltas_action.csv")))


if __name__ == "__main__":
    main()
