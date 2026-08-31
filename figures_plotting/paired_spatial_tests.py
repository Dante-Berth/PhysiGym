#!/usr/bin/env python
"""Paired sign tests along the SPATIAL axis, at matched (content, seed, direction).

Why this exists.  Ch. 6 reports family comparisons with n = 4 observation modes
per family, where the Mann-Whitney U test bottoms out at p = 0.029 regardless of
how large the effect is (ch6_results.tex, "Three honest caveats").  The modes are
also not independent: by construction they share observation content.

The fix is to pair.  Three image modes carry the SAME observation content as a
spatial-scalar mode and differ only in whether position is resolved or averaged
into a k=1 centroid summary.  Comparing within such a pair, at a matched seed and
transfer direction, isolates spatial resolution with everything else held fixed,
and turns 4-vs-4 modes into a set of paired differences a sign test can use.

Pairs (scalar, image), all sharing cell content:
    P1  S3s  <-> I2    cells + substrates          (clean: no m1m2 on either side)
    P2  S3m  <-> I1m   cells + M1/M2
    P3  S3sm <-> I2m   cells + substrates + M1/M2

CAVEAT, reported and not hidden: the families implement the m1m2 split
differently (ch5_mdp.tex:483-487).  Spatial-scalars REPLACE the macrophage
descriptor with M1 and M2 descriptors; images ADD M1/M2 maps beside the combined
channel.  So P2 and P3 carry a small content asymmetry leaning towards the image
side.  P1 has no m1m2 on either side and is therefore the clean pair; the script
reports P1 alone as well as the aggregate so the reader can check that the
conclusion does not rest on the asymmetry.

Run selection goes through analyse_q_calibration.select_runs() -- mandatory.
Globbing the per-mode CSVs directly admits crashed relaunches and double-counts
seeds, which silently erases the family separation (CRITIC_METRICS_EXPLAINED.md).

Emits out_paired_spatial/{paired_deltas.csv, tab_paired_spatial.tex}.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from analyse_q_calibration import (
    select_runs, _seed_id, _tail, MODE_ID, DIRECTIONS,
)
# The uniform-action arm (H6) needs its own selector: the full sweeps live in
# different projects and carry crashed stubs that MIN_ROWS drops.
from plot_action_mode_ablation import seed_files

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out_paired_spatial")
os.makedirs(OUT, exist_ok=True)

B_BOOT = 10_000          # matches plot_tme_bootstrap.py
RNG = np.random.default_rng(0)

# (label, scalar mode, image mode, shared content)
PAIRS = [
    ("P1", "spatial_scalars_cells_substrates",
           "img_mc_cells_substrates",            "cells + substrates"),
    ("P2", "spatial_scalars_cells_m1m2",
           "img_mc_cells_m1m2",                  "cells + M1/M2"),
    ("P3", "spatial_scalars_cells_substrates_m1m2",
           "img_mc_cells_substrates_m1m2",       "cells + substrates + M1/M2"),
]

# metric -> (column, tail length, log-scale?, H1 direction on delta = img - scal)
# tail 50 on the training distribution, 20 on held-out, per the chapter's rule.
METRICS = {
    "test_return":  dict(col="test_return",  tail=20, log=False, h1="greater"),
    "train_return": dict(col="train_return", tail=50, log=False, h1="two-sided"),
    "gap":          dict(col=None,           tail=None, log=False, h1="less"),
    "td_residual":  dict(col="qf1_loss",     tail=20, log=True,  h1="less"),
    "q_mae":        dict(col="q_mae",        tail=20, log=True,  h1="less"),
}


def endpoint(direction, mode, col, tail):
    """{seed: end-of-training value} for one (direction, mode, column)."""
    return {_seed_id(f): _tail(f, col, tail) for f in select_runs(direction, mode)}


def endpoint_action(direction, action_mode, mode, col, tail):
    """As endpoint(), but for a given action condition (targeted / full)."""
    return {_seed_id(f): _tail(f, col, tail)
            for f in seed_files(direction, action_mode, mode)}


def collect_action():
    """H6: the same paired deltas under targeted vs uniform ('full') action.

    Only the two critic metrics are collected, because H6 is about which of them
    survives removing the spatial degree of freedom from the action.
    """
    rows = []
    for direction in DIRECTIONS:
        for action_mode in ("targeted", "full"):
            for tag, scal, img, content in PAIRS:
                for name in ("td_residual", "q_mae"):
                    spec = METRICS[name]
                    sv = endpoint_action(direction, action_mode, scal,
                                         spec["col"], spec["tail"])
                    iv = endpoint_action(direction, action_mode, img,
                                         spec["col"], spec["tail"])
                    for seed in sorted(set(sv) & set(iv)):
                        a, b = iv[seed], sv[seed]
                        if not (np.isfinite(a) and np.isfinite(b)):
                            continue
                        if a <= 0 or b <= 0:
                            continue
                        delta = (0.5 * (np.log(a) - np.log(b))
                                 if name == "td_residual" else np.log(a) - np.log(b))
                        rows.append(dict(metric=name, pair=tag, action=action_mode,
                                         direction=direction, seed=seed,
                                         image=a, scalar=b, delta=delta))
    return pd.DataFrame(rows)


def collect():
    """One row per (metric, pair, direction, seed) with the paired delta."""
    rows = []
    for direction in DIRECTIONS:
        for tag, scal, img, content in PAIRS:
            # gap = train - test, so it needs both tails before differencing.
            cache = {}
            for side, mode in (("scal", scal), ("img", img)):
                cache[(side, "test")] = endpoint(direction, mode, "test_return", 20)
                cache[(side, "train")] = endpoint(direction, mode, "train_return", 50)

            for name, spec in METRICS.items():
                if name == "gap":
                    sv = {s: cache[("scal", "train")][s] - cache[("scal", "test")][s]
                          for s in cache[("scal", "train")]}
                    iv = {s: cache[("img", "train")][s] - cache[("img", "test")][s]
                          for s in cache[("img", "train")]}
                else:
                    sv = endpoint(direction, scal, spec["col"], spec["tail"])
                    iv = endpoint(direction, img, spec["col"], spec["tail"])

                # strict seed matching: only seeds present on BOTH sides
                for seed in sorted(set(sv) & set(iv)):
                    a, b = iv[seed], sv[seed]
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    if spec["log"]:
                        if a <= 0 or b <= 0:
                            continue
                        # sqrt(L) per the chapter, i.e. half the log of the loss
                        delta = 0.5 * (np.log(a) - np.log(b)) if name == "td_residual" \
                                else np.log(a) - np.log(b)
                    else:
                        delta = a - b
                    rows.append(dict(metric=name, pair=tag, content=content,
                                     direction=direction, seed=seed,
                                     image=a, scalar=b, delta=delta))
    return pd.DataFrame(rows)


def boot_ci(x, b=B_BOOT):
    x = np.asarray(x, float)
    means = RNG.choice(x, size=(b, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def sign_test(deltas, h1):
    """Binomial sign test.  Zeros are dropped, which is the conservative choice."""
    d = np.asarray(deltas, float)
    d = d[d != 0]
    n = len(d)
    k = int((d > 0).sum())          # successes = deltas above zero
    if h1 == "greater":
        p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
        consistent = k
    elif h1 == "less":
        p = stats.binomtest(n - k, n, 0.5, alternative="greater").pvalue
        consistent = n - k
    else:
        p = stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
        consistent = k
    return consistent, n, p


def rank_biserial(d):
    """Matched-pairs effect size: r = (W+ - W-) / (n(n+1)/2), the signed-rank sum
    normalised by its maximum.  +1 means every pair favours the image side, -1
    every pair the scalar side, 0 an even split by rank mass."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / (n * (n + 1) / 2)


def summarise(df, subset=None, label=""):
    out = []
    for name, spec in METRICS.items():
        sub = df[df.metric == name]
        if subset is not None:
            sub = sub[subset(sub)]
        if sub.empty:
            continue
        d = sub.delta.to_numpy()
        k, n, p = sign_test(d, spec["h1"])
        lo, hi = boot_ci(d)
        out.append(dict(scope=label, metric=name, prediction=spec["h1"],
                        consistent=k, n=n, mean=d.mean(), lo=lo, hi=hi,
                        p=p, r=rank_biserial(d)))
    return pd.DataFrame(out)


PRED = {"greater": r"$\Delta > 0$", "less": r"$\Delta < 0$",
        "two-sided": r"$\Delta = 0$"}
LABEL = {"test_return": "Held-out return", "train_return": "Training return",
         "gap": "Generalisation gap", "td_residual": r"TD residual $\sqrt{\mathcal{L}}$",
         "q_mae": r"Calibration $m$"}


def fmt_p(p):
    if p >= 0.01:
        return f"${p:.2f}$"
    e = int(np.floor(np.log10(p)))
    return f"${p/10**e:.0f} \\times 10^{{{e}}}$"


def to_tex(summary, path):
    lines = [
        r"% Generated by paired_spatial_tests.py -- do not edit by hand.",
        r"\begin{tabular}{lccrc}", r"  \toprule",
        r"  \textbf{Metric} & \textbf{Prediction} & \textbf{Sign consistency}"
        r" & \textbf{Mean $\Delta$ [95\% CI]} & $p$ \\", r"  \midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"  {LABEL[r.metric]} & {PRED[r.prediction]} & ${r.consistent}/{r.n}$ & "
            f"${r['mean']:+.2f}$ [${r.lo:+.2f}$, ${r.hi:+.2f}$] & {fmt_p(r.p)} \\\\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    df = collect()
    df.to_csv(os.path.join(OUT, "paired_deltas.csv"), index=False)

    print("== pairs available, per (pair, direction) ==")
    cnt = (df[df.metric == "test_return"]
           .groupby(["pair", "direction"]).seed.apply(lambda s: sorted(s)))
    for (pair, direction), seeds in cnt.items():
        print(f"  {pair} {direction:9} seeds={seeds} n={len(seeds)}")
    n_tot = (df.metric == "test_return").sum()
    print(f"  TOTAL matched pairs on held-out return: {n_tot}")

    allsum = summarise(df, label="all")
    print("\n== paired sign tests, all three content pairs ==")
    for _, r in allsum.iterrows():
        print(f"  {LABEL[r.metric]:28} {PRED[r.prediction]:14} "
              f"{r.consistent:2}/{r.n:<3} mean={r['mean']:+8.3f} "
              f"[{r.lo:+.3f},{r.hi:+.3f}] p={r.p:.2e} r={r.r:+.2f}")

    p1 = summarise(df, subset=lambda s: s.pair == "P1", label="P1 only")
    print("\n== P1 alone (S3s vs I2, the content-clean pair) ==")
    for _, r in p1.iterrows():
        print(f"  {LABEL[r.metric]:28} {r.consistent:2}/{r.n:<3} "
              f"mean={r['mean']:+8.3f} p={r.p:.3f}")

    print("\n== by direction ==")
    for d in DIRECTIONS:
        s = summarise(df, subset=lambda x, d=d: x.direction == d, label=d)
        for _, r in s.iterrows():
            print(f"  {d:9} {LABEL[r.metric]:28} {r.consistent:2}/{r.n:<3} "
                  f"mean={r['mean']:+8.3f} p={r.p:.3f}")

    # ---- H6: does the paired effect survive removing spatial action? ----
    act = collect_action()
    act.to_csv(os.path.join(OUT, "paired_deltas_action.csv"), index=False)
    print("\n== H6: paired effect by action condition ==")
    for name in ("td_residual", "q_mae"):
        for am in ("targeted", "full"):
            sub = act[(act.metric == name) & (act.action == am)]
            if sub.empty:
                continue
            k, n, p = sign_test(sub.delta.to_numpy(), METRICS[name]["h1"])
            lo, hi = boot_ci(sub.delta.to_numpy())
            print(f"  {LABEL[name]:28} {am:9} {k:2}/{n:<3} "
                  f"mean={sub.delta.mean():+7.3f} [{lo:+.3f},{hi:+.3f}] p={p:.2e}")

    to_tex(allsum, os.path.join(OUT, "tab_paired_spatial.tex"))
    pd.concat([allsum, p1]).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(f"\nwrote {OUT}/tab_paired_spatial.tex")
