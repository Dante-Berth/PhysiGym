"""Value calibration over training under uniform action (old-model sweep).

Same methodology as plot_q_calibration.py (EWMA-20 per seed, mean over five seeds
+- 1 sd; the TD loss in log space), restricted to the three modes of the uniform
ablation.  Uniform action is drawn solid; the same mode under targeted action is
drawn dotted and fainter, as the reference.  Row 1: signed calibration error
test_q_bias on held-out episodes.  Row 2: qf1_loss on the replay buffer.
Outputs out_q_calibration/q_bias_uniform.pdf/.png.  Run from figures_plotting/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyse_q_calibration import _ewma, select_runs
from plot_action_mode_ablation import seed_files

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_q_calibration")
os.makedirs(OUT, exist_ok=True)
AR = 6
MODES = [("scalars_macrophages", "S1", "#5c677d", "o"),
         ("spatial_scalars_cells_m1m2", "S3m", "#0077b6", "s"),
         ("img_mc_cells_substrates", "I2", "#c1121f", "D")]
DIRS = [("rect2net", r"rectangle $\rightarrow$ network-field"),
        ("net2rect", r"network-field $\rightarrow$ rectangle")]


def series(files, col):
    out = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in ("step", col))
        if col not in d.columns:
            continue
        d = d[["step", col]].dropna().sort_values("step")
        if len(d) > 10:
            out.append((d.step.to_numpy(), _ewma(d[col].to_numpy(), 20)))
    return out


def curve(ser, log):
    lo = max(s[0] for s, _ in ser); hi = min(s[-1] for s, _ in ser)
    g = np.linspace(lo, hi, 300)
    m = np.array([np.interp(g, s, v) for s, v in ser])
    if log:
        l = np.log10(np.clip(m, 1e-12, None)); mu, sd = l.mean(0), l.std(0)
        return g * AR / 1e5, 10 ** mu, 10 ** (mu - sd), 10 ** (mu + sd)
    return g * AR / 1e5, m.mean(0), m.mean(0) - m.std(0), m.mean(0) + m.std(0)


plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True)
for j, (d, title) in enumerate(DIRS):
    for i, (col, log) in enumerate([("q_bias", False), ("qf1_loss", True)]):
        ax = axes[i][j]
        for mode, name, c, mk in MODES:
            for act, ls, lw, al in (("targeted", ":", 1.3, 0.55), ("full", "-", 2.0, 1.0)):
                files = (select_runs(d, mode) if act == "targeted"
                         else seed_files(d, "full", mode))
                ser = series(files, col)
                if not ser:
                    continue
                x, mu, lo, hi = curve(ser, log)
                ax.plot(x, mu, color=c, ls=ls, lw=lw, alpha=al, zorder=3 if act == "full" else 2,
                        label=f"{name}, {'uniform' if act == 'full' else 'targeted'}")
                if act == "full":
                    ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0, zorder=1)
        if log:
            ax.set_yscale("log"); ax.set_ylabel(r"TD loss $\mathcal{L}(Q_{\phi_1})$ (replay buffer)")
        else:
            ax.axhline(0, color="k", lw=0.8, alpha=0.6); ax.set_ylabel(r"signed error $b$ (held-out)")
        if i == 0:
            ax.set_title(title, fontsize=10.5)
        if i == 1:
            ax.set_xlabel(r"cumulative simulator steps ($\times 10^5$)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, ncol=6, loc="lower center", frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.05))
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"q_bias_uniform.{e}"))
print("wrote q_bias_uniform")
