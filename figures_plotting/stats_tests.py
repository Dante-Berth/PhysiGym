#!/usr/bin/env python
"""Explicit significance tests / effect sizes for the observation-space study,
complementing the bootstrap CIs (addresses the 'stats beyond CIs' gap).

Reuses the exact per-seed end-of-training returns that feed plot_tme_new.py
(n=5 per mode, first-5-by-seed-id via its MAX_SEEDS cap). No new runs.

Outputs numbers used verbatim in Results/Discussion:
  1. Image vs spatial-scalar OOD test gap: Mann-Whitney U + Cliff's delta.
  2. M1/M2-split ablation (I1 vs I1m, I2 vs I2m): paired diff + TOST equivalence.
  3. Seed-variance claim: Levene test on across-seed test spread (image vs scalar).
"""
import numpy as np
from scipy import stats
from plot_tme_new import load, _ewma

IMG = ["img_mc_cells", "img_mc_cells_substrates", "img_mc_cells_substrates_m1m2"]
SCAL = ["spatial_scalars_cells_substrates", "spatial_scalars_cells_m1m2",
        "spatial_scalars_cells_substrates_m1m2",
        "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2"]


def per_seed_test(mode):
    """End-of-training EWMA-50 test return, one value per seed (n=5)."""
    return np.array([_ewma(v)[-20:].mean() for _, v in load(mode, "test_return")])


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def tost(a, b, margin):
    """Two one-sided tests for equivalence of means within +-margin (Welch)."""
    diff = np.mean(a) - np.mean(b)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    df = len(a) + len(b) - 2
    t_low = (diff + margin) / se     # H0: diff <= -margin
    t_high = (diff - margin) / se    # H0: diff >= +margin
    p_low = stats.t.sf(t_low, df)
    p_high = stats.t.cdf(t_high, df)
    return diff, max(p_low, p_high)  # equivalence p = max of the two one-sided


if __name__ == "__main__":
    img = {m: per_seed_test(m) for m in IMG}
    scal = {m: per_seed_test(m) for m in SCAL}

    print("== 1. Image vs spatial-scalar OOD test return ==")
    A = np.concatenate(list(img.values()))
    B = np.concatenate(list(scal.values()))
    U, p = stats.mannwhitneyu(A, B, alternative="greater")
    print(f"  image n={len(A)} mean={A.mean():.1f}; scalar n={len(B)} mean={B.mean():.1f}")
    print(f"  Mann-Whitney U={U:.0f}, one-sided p={p:.2e}")
    print(f"  Cliff's delta={cliffs_delta(A, B):+.3f} (1.0 = complete separation)")

    print("\n== 2. M1/M2-split ablation, TOST equivalence (margin=10 ~= 1 sigma) ==")
    for base, m1m2 in [("img_mc_cells", "img_mc_cells_m1m2"),
                       ("img_mc_cells_substrates", "img_mc_cells_substrates_m1m2")]:
        a = per_seed_test(base); b = per_seed_test(m1m2)
        _, pw = stats.ttest_ind(a, b, equal_var=False)
        diff, ptost = tost(a, b, margin=10.0)
        print(f"  {base} ({a.mean():+.1f}) vs {m1m2} ({b.mean():+.1f}): "
              f"diff={diff:+.1f}, Welch p={pw:.2f}, TOST p={ptost:.3f} "
              f"({'EQUIVALENT' if ptost < 0.05 else 'not shown equivalent'})")

    print("\n== 3. Seed-variance: Levene (image vs scalar test spread) ==")
    W, pv = stats.levene(A, B, center="median")
    print(f"  Levene W={W:.2f}, p={pv:.2f} "
          f"({'no variance difference' if pv > 0.05 else 'variance differs'})")
