#!/usr/bin/env python
"""Forest plot of the paired spatial-axis effects (Ch. 6 §6.5).

One panel per metric.  Each row is a (content pair, transfer direction) cell:
the mean paired delta with a 95% bootstrap CI, the per-seed deltas behind it as
light scatter, and a bold aggregate row with sign consistency and p.  A dashed
line marks zero and each panel is annotated with the prediction it tests.

Reads out_paired_spatial/paired_deltas.csv, written by paired_spatial_tests.py.
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


def main():
    df = pd.read_csv(os.path.join(OUT, "paired_deltas.csv"))
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))

    for ax, metric in zip(axes.ravel(), PANELS):
        sub = df[df.metric == metric]
        spec = METRICS[metric]

        rows, y = [], 0.0
        for tag, _, _, content in PAIRS:
            for direction in ("rect2net", "net2rect"):
                cell = sub[(sub.pair == tag) & (sub.direction == direction)]
                if cell.empty:
                    continue
                rows.append((y, f"{tag}  {DIR_SHORT[direction]}", cell.delta.to_numpy()))
                y += 1.0
            y += 0.35                              # gap between content pairs

        for yy, label, d in rows:
            m = d.mean()
            # one colour per row: blue where the row's mean runs with the
            # prediction, red where it runs against it.
            favours = (m < 0) if spec["h1"] == "less" else (m > 0)
            c = NEG if favours else POS
            ax.scatter(d, np.full_like(d, yy), s=14, alpha=0.35, color=c,
                       zorder=2, linewidths=0)
            lo, hi = boot_ci(d)
            ax.plot([lo, hi], [yy, yy], color=c, lw=1.8, zorder=3, solid_capstyle="round")
            ax.plot([m], [yy], "o", color=c, ms=6, zorder=4)

        # aggregate row
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
        ax.set_xlabel(XLABEL[metric])
        ax.set_title(f"Prediction: {PRED[spec['h1']]}", fontsize=9, loc="left")
        ax.margins(y=0.06)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"paired_spatial.{ext}"))
    print(f"wrote {OUT}/paired_spatial.pdf")


if __name__ == "__main__":
    main()
