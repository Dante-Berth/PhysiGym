"""Build the episode-comparison figure from replay rollouts on a shared IC.

Reads {I2,S3s,POMDP}_replay.csv from episode_rollouts/ and produces a 3-panel
figure (cumulative reward, tumor cell count, treatment action) mirroring the
paper's fig_episode_comparison. All three ran the SAME network-field initial
condition (ic_episode_A.csv, 124 tumor cells) so only the observation mode
differs.

The treatment-action panel shows a rolling mean (window=6, matching the
action_repeat=6 used in training) of the raw per-step dose action, since the
raw action is genuinely noisy step-to-step (a real property of the policy,
not a plotting artifact) and unreadable at full resolution.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(BASE, "episode_rollouts")
OUT = os.path.join(BASE, "out_tme_new")
os.makedirs(OUT, exist_ok=True)

MODES = {
    "I2":    dict(label="I2  img_mc_cells_substrates", color="#e63946", lw=2.0),
    "S3s":   dict(label="S3s  spatial_scalars_cells_substrates", color="#4361ee", lw=1.7),
    "POMDP": dict(label="POMDP  scalars_macrophages", color="#adb5bd", lw=1.6),
}


def load(mode):
    f = os.path.join(IN, f"{mode}_replay.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def main():
    dfs = {m: load(m) for m in MODES}
    dfs = {m: d for m, d in dfs.items() if d is not None and len(d) > 1}
    if not dfs:
        print("no rollout CSVs found in", IN)
        return

    ROLL_WINDOW = 6  # matches action_repeat=6 used in training

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
    summary = []
    for m, d in dfs.items():
        meta = MODES[m]
        axes[0].plot(d["step"], d["cum_reward"], color=meta["color"], lw=meta["lw"], label=meta["label"])
        axes[1].plot(d["step"], d["number_tumor"], color=meta["color"], lw=meta["lw"])
        dose_smooth = d["action_dose"].rolling(window=ROLL_WINDOW, center=True, min_periods=1).mean()
        axes[2].plot(d["step"], dose_smooth, color=meta["color"], lw=meta["lw"], alpha=0.9)
        summary.append((m, int(d["number_tumor"].iloc[-1]),
                        float(d["cum_reward"].iloc[-1]), float(d["cum_dose"].iloc[-1])))

    axes[0].set_ylabel("cumulative reward")
    axes[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    axes[0].legend(fontsize=8, loc="best")
    axes[1].set_ylabel("tumor cell count")
    axes[2].set_ylabel(f"treatment action\n(dose, rolling mean w={ROLL_WINDOW})")
    axes[2].set_xlabel("gym step")
    fig.suptitle("Matched network-field episode (identical IC, 124 tumor cells)", fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_episode_comparison.pdf"))
    plt.close(fig)

    print("=== end-of-episode summary ===")
    print(f"{'mode':7s} {'final_tumor':>11s} {'cum_return':>11s} {'cum_dose':>9s}")
    for m, t, r, dz in summary:
        print(f"{m:7s} {t:11d} {r:11.1f} {dz:9.1f}")
    with open(os.path.join(OUT, "episode_summary.txt"), "w") as f:
        for m, t, r, dz in summary:
            f.write(f"{m}: final_tumor={t}, cum_return={r:.1f}, cum_dose={dz:.1f}\n")
    print("wrote", os.path.join(OUT, "fig_episode_comparison.pdf"))


if __name__ == "__main__":
    main()
