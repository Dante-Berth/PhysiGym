"""Per-mode train/test return AND value calibration, for both transfer directions.

Aggregation follows plot_tme_new.py exactly so the return columns reproduce the
already-reported table: EWMA-50, mean of the last 50 (train) / last 20 (test)
smoothed points, n=5 lowest seed-ids per mode.

The Q columns use the test convention (last 20), because test_q_bias/test_q_mae
are computed per *test* episode: each test episode is replayed, the discounted
Monte-Carlo return computed per step, and min(qf1,qf2) evaluated on the stored
(obs, action) pairs (run.py:1198-1249).

    q_bias = mean(Q - MC)   signed; > 0 means the critic overestimates value
    q_mae  = mean|Q - MC|   magnitude of miscalibration

There is no train-side calibration metric, so only OOD calibration is available
-- not a train-vs-test calibration contrast.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

BASE = os.path.dirname(os.path.abspath(__file__))

DIRECTIONS = {
    "rect2net": dict(
        dir="wandb_train_rectangle_test_networkfield",
        label="train = rectangle, test = network-field (EWRL-reported sweep)",
        # restrict to the 59 runs that produced the reported numbers
        restrict="wandb_tme_new/manifest.csv",
    ),
    "net2rect": dict(
        dir="wandb_train_networkfield_test_rectangle",
        label="train = network-field, test = rectangle (reverse transfer, T2)",
        restrict=None,
    ),
}

MODE_ID = {
    "img_mc_cells": "I1",
    "img_mc_cells_m1m2": "I1m",
    "img_mc_cells_substrates": "I2",
    "img_mc_cells_substrates_m1m2": "I2m",
    "spatial_scalars_cells_substrates": "S3s",
    "spatial_scalars_cells_substrates_m1m2": "S3sm",
    "spatial_scalars_cells_m1m2": "S3m",
    "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2": "S5m",
    "scalars_macrophages": "POMDP",
    "random_baseline": "RAND",
}
FAMILY = {m: ("image" if m.startswith("img") else
              "scalar" if m.startswith("spatial_scalars") else
              "pomdp" if m == "scalars_macrophages" else "random")
          for m in MODE_ID}

MAX_SEEDS = 5


def _ewma(a, w=50):
    a = np.asarray(a, float)
    return a if len(a) == 0 else uniform_filter1d(a, size=min(w, len(a)), mode="nearest")


def _seed_id(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def _tail(path, col, n):
    """Mean of the last n smoothed points of one column, or NaN if too sparse."""
    df = pd.read_csv(path, usecols=lambda c: c in ("step", col))
    if col not in df.columns:
        return np.nan
    v = df[["step", col]].dropna().sort_values("step")[col].to_numpy()
    return np.nan if len(v) <= 3 else _ewma(v)[-n:].mean()


def select_runs(direction, mode, max_seeds=MAX_SEEDS):
    """The canonical run selection for a (direction, mode).  THE single source of
    truth -- every script, table and figure in Ch. 6 must call this rather than
    re-deriving it, or the figures and the tables drift apart (they did: see
    RESEARCH_LOG.md 2026-08-21).

    Two filters, in this order:

    1.  Manifest restriction, where the direction defines one.  rect2net holds
        runs beyond the reported sweep; only the run-ids in wandb_tme_new
        produced the numbers this chapter reports.
    2.  De-duplicate by seed BEFORE taking the first five.  Some modes hold more
        than one run per seed (a relaunch after a crash), and a plain [:5] then
        returns e.g. seeds 1,1,2,2,3 -- three distinct seeds double-counted and
        reported as n=5, which both biases the mean towards the repeated seeds
        and understates sigma.  First run-id per seed, so deterministic.

    Together these also drop the crashed relaunches, which is why every run this
    returns reaches at least 94.5k of the 100k agent decisions -- selecting
    without them admits runs that stop at 5k and truncates any figure that
    clips its x-axis to the shortest series.
    """
    cfg = DIRECTIONS[direction]
    prefix = "RANDOM_" if mode == "random_baseline" else "SAC_"
    files = sorted(glob.glob(os.path.join(BASE, cfg["dir"], mode, prefix + "*.csv")),
                   key=_seed_id)
    if cfg["restrict"]:
        keep = set(pd.read_csv(os.path.join(BASE, cfg["restrict"])).run_id)
        files = [f for f in files
                 if os.path.basename(f).rsplit("_", 1)[-1][:-4] in keep]
    seen, dedup = set(), []
    for f in files:
        s = _seed_id(f)
        if s not in seen:
            seen.add(s)
            dedup.append(f)
    return dedup[:max_seeds]


def rows_for(direction):
    out = []
    for mode, mid in MODE_ID.items():
        files = select_runs(direction, mode)
        if not files:
            continue

        stat = {c: np.array([_tail(f, c, n) for f in files])
                for c, n in [("train_return", 50), ("test_return", 20),
                             ("q_bias", 20), ("q_mae", 20),
                             ("qf1_loss", 20), ("qf2_loss", 20)]}
        fin = lambda c: stat[c][~np.isnan(stat[c])]

        r = dict(id=mid, mode=mode, family=FAMILY[mode], n=len(files))
        for c in stat:
            v = fin(c)
            r[f"{c}_mu"] = v.mean() if len(v) else np.nan
            r[f"{c}_sd"] = v.std() if len(v) else np.nan
        r["n_q"] = len(fin("q_bias"))
        r["gap"] = r["train_return_mu"] - r["test_return_mu"]
        out.append(r)
    return pd.DataFrame(out)


def show(direction):
    cfg = DIRECTIONS[direction]
    df = rows_for(direction).sort_values(["family", "id"])
    print("=" * 100)
    print(f"{direction}:  {cfg['label']}")
    print("=" * 100)
    print(f"{'ID':<6}{'family':<8}{'n':>3}{'nq':>4}"
          f"{'train':>9}{'±':>7}{'test':>9}{'±':>7}{'gap':>8}"
          f"{'q_bias':>10}{'±':>8}{'q_mae':>9}{'±':>8}")
    for _, r in df.iterrows():
        f = lambda x: "    n/a" if pd.isna(x) else f"{x:+7.1f}"
        g = lambda x: "   n/a" if pd.isna(x) else f"{x:6.1f}"
        print(f"{r['id']:<6}{r['family']:<8}{r['n']:>3}{r['n_q']:>4}"
              f"{f(r['train_return_mu'])}{g(r['train_return_sd'])}"
              f"{f(r['test_return_mu'])}{g(r['test_return_sd'])}"
              f"{f(r['gap'])}"
              f"{f(r['q_bias_mu']):>10}{g(r['q_bias_sd'])}"
              f"{f(r['q_mae_mu']):>9}{g(r['q_mae_sd'])}")
    print()
    return df


if __name__ == "__main__":
    out = os.path.join(BASE, "out_q_calibration")
    os.makedirs(out, exist_ok=True)
    for d in DIRECTIONS:
        show(d).to_csv(os.path.join(out, f"summary_{d}.csv"), index=False)
    print(f"written to {out}/")
