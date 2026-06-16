"""
plot_initial_conditions.py
──────────────────────────
Generates a 3-panel figure showing the three spatial distributions used
in the EWRL 2026 experiments, built directly from real IC CSVs.

Exact source files:
  network-field (training):
    PhysiCell/data/TME_V2_32_img_mc_cells_1779954610/env0/train/episodes/run_000013/ic_000013.csv
  rectangle (test / OOD):
    PhysiCell/data/TME_V2_32_img_mc_cells_1779954610/env0/test/episodes/run_000004/ic_000004.csv
  circular (test / OOD):
    PhysiCell/data/TME_V2_32_img_mc_cells_1779954610/env0/test/episodes/run_000008/ic_000008.csv

Usage:
    python plot_initial_conditions.py \
        --data_dir PhysiCell/data \
        --out_dir  PhysiGym/paper/ewrl_2026_physigym/img
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent.parent / "PhysiCell" / "data"
DEFAULT_OUT  = SCRIPT_DIR.parent / "paper" / "ewrl_2026_physigym" / "img"

# ── cell colours ──────────────────────────────────────────────────────────────
COLORS = {"tumor": "#555555", "t_cell": "#e63946", "macrophage": "#2171b5"}
LABELS = {"tumor": "Tumor", "t_cell": "T-cell", "macrophage": "Macrophage"}
MARKER_SIZE = 18   # pt²

# ── exact IC CSV paths relative to data_dir ───────────────────────────────────
IC_SPECS = {
    "network_field": {
        "title": "Network-field\n(training distribution)",
        "csv":   "TME_V2_32_img_mc_cells_1779954610/env0/train/episodes/run_000013/ic_000013.csv",
    },
    "rectangle": {
        "title": "Rectangle\n(test, out-of-distribution)",
        "csv":   "TME_V2_32_img_mc_cells_1779954610/env0/test/episodes/run_000004/ic_000004.csv",
    },
    "circular": {
        "title": "Circular\n(test, out-of-distribution)",
        "csv":   "TME_V2_32_img_mc_cells_1779954610/env0/test/episodes/run_000008/ic_000008.csv",
    },
}


def _scatter_panel(ax, df: pd.DataFrame, title: str, domain_size: float = 63.0):
    for cell_type in ("tumor", "t_cell", "macrophage"):
        sub = df[df["type"] == cell_type]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["x"], sub["y"],
            s=MARKER_SIZE,
            c=COLORS[cell_type],
            alpha=0.80,
            linewidths=0,
            label=LABELS[cell_type],
            zorder=2,
            rasterized=True,
        )
    # domain box
    pad = 3.0
    ax.set_xlim(-pad, domain_size + pad)
    ax.set_ylim(-pad, domain_size + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#888888")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))

    for ax, (layout_key, spec) in zip(axes, IC_SPECS.items()):
        csv_path = data_dir / spec["csv"]
        if not csv_path.exists():
            print(f"  WARNING: IC csv not found: {csv_path}")
            ax.text(0.5, 0.5, "IC not found", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_title(spec["title"], fontsize=9)
            continue

        df = pd.read_csv(csv_path)
        counts = df["type"].value_counts().to_dict()
        print(f"  {layout_key}: {csv_path.name}  counts={counts}")
        _scatter_panel(ax, df, spec["title"])

    # shared legend below
    handles = [
        mpatches.Patch(color=COLORS[ct], label=LABELS[ct])
        for ct in ("tumor", "t_cell", "macrophage")
    ]
    # legend with filled circles matching the scatter markers
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color=COLORS[ct], marker='o', linestyle='None',
                      markersize=7, label=LABELS[ct])
        for ct in ("tumor", "t_cell", "macrophage")
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "Initial cell configurations: training vs. held-out test distributions",
        fontsize=9.5,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("pdf", "png"):
        out = out_dir / f"ic_distributions.{ext}"
        plt.savefig(out)
        print(f"  saved → {out}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
