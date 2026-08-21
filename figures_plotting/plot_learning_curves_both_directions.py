"""Learning curves (cumulative return vs. simulator steps), both transfer directions.

Produces the figure Ch. 6 §6.2 asks for ("Still to add here: training curves"),
for all nine observation modes plus the random baseline -- i.e. every row of
tab:results:performance / tab:results:reverse.

One output, a 2x2 panel (rows = transfer direction, cols = train/test):

  learning_curves_both_directions.pdf
      charts/train_return_mean and charts/test_return_mean against cumulative
      simulator steps.  EWMA-50 per seed, then the across-seed mean with a 95%
      bootstrap CI band (B=10,000 seed resamples, percentile method).  The
      right-hand endpoints reproduce the mu columns of Tables 5.1 / 5.3.

"cumulative return" here means what tab:results:performance's caption means by it
-- the return of an episode, the sum of its rewards.  An earlier draft also
emitted the *integral* of these curves over the step axis (area under the
learning curve) under the same name; it was dropped 2026-08-21 because two
different quantities were sharing one term.  If sample efficiency is ever wanted,
integrate these curves -- do not call the result a cumulative return.

AGGREGATION IS DELIBERATELY IDENTICAL TO analyse_q_calibration.py, so the figure
and the tables cannot drift apart:
  - EWMA-50 (uniform_filter1d, mode="nearest") per seed before any averaging
  - n = 5 seeds per mode, the five lowest seed-ids
  - rect2net is restricted to the run-ids in wandb_tme_new/manifest.csv, which is
    the sweep the reported numbers came from

ONE DELIBERATE DEVIATION, and it is a fix.  analyse_q_calibration.py takes
sorted(files)[:5] by seed-id.  In the net2rect direction random_baseline/ holds
*ten* runs -- two per seed, seeds 1-5 twice over -- so that slice returns seeds
1,1,2,2,3: three distinct seeds, two of them double-counted, reported as n=5.
The same happens to rect2net RAND (1,1,2,2,3) and rect2net S3m (1,1,2,3,4).
Here seeds are de-duplicated before the slice (first run-id per seed, so the
choice is deterministic), giving five genuinely distinct seeds.  Exactly three
rows move -- RAND in both directions and S3m in rect2net; every other row of both
tables is bit-identical before and after.  analyse_q_calibration.py carries the
same fix; RESEARCH_LOG.md 2026-08-21 has the before/after numbers.

Run from figures_plotting/; writes to out_learning_curves/.
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
from scipy.ndimage import uniform_filter1d

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out_learning_curves")
os.makedirs(OUT, exist_ok=True)

ACTION_REPEAT = 6      # each agent decision is held for 6 simulator steps
MAX_SEEDS = 5
EWMA_W = 50
B = 10_000             # bootstrap resamples
CI = (2.5, 97.5)
NGRID = 400
STEP_MAX = 100_000     # agent decisions; x 6 = 6e5 simulator steps
RNG = np.random.default_rng(0)

DIRECTIONS = {
    "rect2net": dict(
        dir="wandb_train_rectangle_test_networkfield",
        restrict="wandb_tme_new/manifest.csv",
        row=r"rectangle $\rightarrow$ network-field",
        train="Train (rectangle)",
        test="Test (network-field, held out)",
    ),
    "net2rect": dict(
        dir="wandb_train_networkfield_test_rectangle",
        restrict=None,
        row=r"network-field $\rightarrow$ rectangle",
        train="Train (network-field)",
        test="Test (rectangle, held out)",
    ),
}

# Ordered least-to-most spatial, matching tab:mdp:obs-modes and the table rows.
# Colour encodes FAMILY, because family separation is what the chapter argues:
# scalar modes cool + dashed, image modes warm + solid, baselines grey/black.
# Within a family the four hues are spread in LIGHTNESS as well as hue, so the
# pairs that differ only by an m1m2 split stay distinguishable in greyscale.
MODE_META = {
    "random_baseline":                dict(id="RAND",  color="#212529", ls=(0, (1, 1.6)), lw=1.4, z=1),
    "scalars_macrophages":            dict(id="POMDP", color="#8d99ae", ls="-.",          lw=1.5, z=2),
    "spatial_scalars_cells_m1m2":     dict(id="S3m",   color="#48cae4", ls="--",          lw=1.6, z=3),
    "spatial_scalars_cells_substrates":       dict(id="S3s",  color="#4361ee", ls="--", lw=1.8, z=4),
    "spatial_scalars_cells_substrates_m1m2":  dict(id="S3sm", color="#3a0ca3", ls="--", lw=1.6, z=3),
    "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2":
                                      dict(id="S5m",   color="#9d4edd", ls="--",          lw=1.6, z=3),
    "img_mc_cells":                   dict(id="I1",    color="#ffb703", ls="-",           lw=1.8, z=5),
    "img_mc_cells_m1m2":              dict(id="I1m",   color="#f3722c", ls="-",           lw=1.6, z=5),
    "img_mc_cells_substrates":        dict(id="I2",    color="#e63946", ls="-",           lw=2.0, z=6),
    "img_mc_cells_substrates_m1m2":   dict(id="I2m",   color="#b5179e", ls="-",           lw=1.8, z=6),
}
MODE_ORDER = list(MODE_META)


def _style():
    plt.rcParams.update({
        "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.22,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _ewma(a, w=EWMA_W):
    a = np.asarray(a, float)
    return a if len(a) == 0 else uniform_filter1d(a, size=min(w, len(a)), mode="nearest")


def _seed_id(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def seed_files(direction, mode):
    """The five run CSVs for one mode: lowest seed-ids, one run per seed."""
    cfg = DIRECTIONS[direction]
    prefix = "RANDOM_" if mode == "random_baseline" else "SAC_"
    files = sorted(glob.glob(os.path.join(BASE, cfg["dir"], mode, prefix + "*.csv")),
                   key=_seed_id)
    if cfg["restrict"]:
        keep = set(pd.read_csv(os.path.join(BASE, cfg["restrict"])).run_id)
        files = [f for f in files
                 if os.path.basename(f).rsplit("_", 1)[-1][:-4] in keep]
    seen, dedup = set(), []
    for f in files:                     # deterministic: first run-id per seed
        s = _seed_id(f)
        if s not in seen:
            seen.add(s)
            dedup.append(f)
    return dedup[:MAX_SEEDS]


def seed_matrix(direction, mode, col, grid):
    """(n_seeds, len(grid)) of EWMA-50 values, NaN outside each seed's own range."""
    rows = []
    for f in seed_files(direction, mode):
        d = pd.read_csv(f, usecols=["step", col]).dropna().sort_values("step")
        if len(d) <= 3:
            continue
        rows.append(np.interp(grid, d["step"].to_numpy(), _ewma(d[col].to_numpy()),
                              left=np.nan, right=np.nan))
    return np.array(rows) if rows else np.empty((0, len(grid)))


def bootstrap_band(mat):
    """Per-column mean and 95% bootstrap-of-the-mean CI over seeds.

    Only columns where EVERY seed has data are returned; elsewhere the band would
    be built from a shrinking, self-selected subset of seeds, which is exactly the
    artefact a CI is supposed to expose rather than hide.
    """
    n, T = mat.shape
    ok = ~np.isnan(mat).any(axis=0)
    mean = np.full(T, np.nan)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    if n == 0 or not ok.any():
        return mean, lo, hi, ok
    sub = mat[:, ok]
    mean[ok] = sub.mean(axis=0)
    if n == 1:
        lo[ok] = hi[ok] = sub[0]
        return mean, lo, hi, ok
    # multinomial counts == resampling seeds with replacement, but O(B*n) memory
    counts = RNG.multinomial(n, np.full(n, 1.0 / n), size=B)      # (B, n)
    boot = (counts @ sub) / n                                     # (B, T_ok)
    lo[ok] = np.percentile(boot, CI[0], axis=0)
    hi[ok] = np.percentile(boot, CI[1], axis=0)
    return mean, lo, hi, ok





# ── panel assembly ───────────────────────────────────────────────────────────
def collect(direction, col, grid):
    return {m: seed_matrix(direction, m, col, grid) for m in MODE_ORDER}





def draw(fname):
    """The 2x2 panel: rows = transfer direction, cols = train / held-out."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True, sharey="all")
    grid = np.linspace(0, STEP_MAX, NGRID)
    x = grid * ACTION_REPEAT / 1e5

    for r, direction in enumerate(DIRECTIONS):
        cfg = DIRECTIONS[direction]
        for c, col in enumerate(["train_return", "test_return"]):
            ax = axes[r, c]
            mats = collect(direction, col, grid)
            for mode in MODE_ORDER:
                m, mat = MODE_META[mode], mats[mode]
                if mat.size == 0:
                    continue
                mean, lo, hi, _ = bootstrap_band(mat)
                ax.plot(x, mean, color=m["color"], ls=m["ls"], lw=m["lw"],
                        zorder=m["z"], label=m["id"])
                ax.fill_between(x, lo, hi, color=m["color"], alpha=0.13,
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

    handles = [Line2D([], [], color=MODE_META[m]["color"], ls=MODE_META[m]["ls"],
                      lw=MODE_META[m]["lw"], label=MODE_META[m]["id"])
               for m in MODE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False,
               bbox_to_anchor=(0.5, -0.035), fontsize=9.5, columnspacing=1.5,
               handlelength=2.4)
    fig.tight_layout(rect=(0.015, 0.015, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"))
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png")


# ── numbers the caption is allowed to quote ──────────────────────────────────
def endpoint_check():
    """Right-hand endpoint of each curve vs. the table convention, per mode.

    The tables average the last 50 (train) / 20 (test) smoothed points per seed;
    the curve endpoint is the interpolated value at step 100,000.  They are not
    the same statistic, so they are printed side by side rather than assumed
    equal -- if they disagree by more than a point or two, one of them is wrong.
    """
    grid = np.linspace(0, STEP_MAX, NGRID)
    rows = []
    for direction in DIRECTIONS:
        for col, tail in [("train_return", 50), ("test_return", 20)]:
            for mode in MODE_ORDER:
                mat = seed_matrix(direction, mode, col, grid)
                if mat.size == 0:
                    continue
                ok = ~np.isnan(mat).any(axis=0)
                curve_end = mat[:, ok][:, -1].mean()
                tails = []
                for f in seed_files(direction, mode):
                    d = pd.read_csv(f, usecols=["step", col]).dropna().sort_values("step")
                    if len(d) > 3:
                        tails.append(_ewma(d[col].to_numpy())[-tail:].mean())
                rows.append(dict(direction=direction, split=col.split("_")[0],
                                 id=MODE_META[mode]["id"], n=mat.shape[0],
                                 curve_end=curve_end, table_mu=np.mean(tails),
                                 delta=curve_end - np.mean(tails)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "endpoint_check.csv"), index=False)
    return df





if __name__ == "__main__":
    draw("learning_curves_both_directions")

    print("\nCurve endpoint vs. table convention (delta should be small):")
    ep = endpoint_check()
    print(ep.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(f"\n  max |delta| = {ep.delta.abs().max():.2f}")
    print("\nOutputs in", OUT)
