"""Every qf1_loss / calibration number Ch.6 sec 6.3 quotes, from one computation.

Per-seed tail statistic (EWMA-50, mean of last 20 points -- the convention
analyse_q_calibration uses for every test-side column), then aggregated over
seeds.  RMSE_TD is per-seed sqrt BEFORE averaging: sqrt of the mean loss is not
the mean of the per-run RMSEs, and the runs are the unit of analysis.
"""
import numpy as np, pandas as pd
from scipy import stats as sps
from analyse_q_calibration import _ewma, select_runs, MODE_ID, FAMILY

def tail(f, col, n=20):
    df = pd.read_csv(f, usecols=lambda c: c in ("step", col))
    if col not in df.columns:
        return np.nan
    v = df[["step", col]].dropna().sort_values("step")[col].to_numpy()
    return np.nan if len(v) <= 3 else _ewma(v)[-n:].mean()

MW = lambda A, B: sps.mannwhitneyu(A, B, alternative="two-sided").pvalue

for d, lab in (("rect2net", "rectangle -> network-field"),
               ("net2rect", "network-field -> rectangle")):
    print("=" * 84)
    print(lab)
    print("=" * 84)
    print(f"{'mode':<7}{'fam':<8}{'qf1_loss':>11}{'RMSE_TD':>10}{'alpha':>8}"
          f"{'m (OOD)':>10}{'m/RMSE_TD':>12}")
    rows = {}
    for mode, mid in MODE_ID.items():
        if mode == "random_baseline":
            continue
        fs = select_runs(d, mode)
        loss = np.array([tail(f, "qf1_loss") for f in fs])
        rmse = np.sqrt(loss)
        m = np.array([tail(f, "q_mae") for f in fs])
        a = np.array([tail(f, "alpha") for f in fs])
        rows[mid] = dict(fam=FAMILY[mode], loss=np.nanmean(loss),
                         rmse=np.nanmean(rmse), m=np.nanmean(m),
                         alpha=np.nanmean(a), ratio=np.nanmean(m / rmse))
        r = rows[mid]
        print(f"{mid:<7}{r['fam']:<8}{r['loss']:>11.1f}{r['rmse']:>10.2f}"
              f"{r['alpha']:>8.2f}{r['m']:>10.1f}{r['ratio']:>12.2f}")
    print()
    for key, name in (("loss", "qf1_loss"), ("rmse", "RMSE_TD"),
                      ("alpha", "alpha"), ("ratio", "m / RMSE_TD")):
        A = np.array([r[key] for r in rows.values() if r["fam"] == "image"])
        B = np.array([r[key] for r in rows.values() if r["fam"] == "scalar"])
        # direction-agnostic: the families overlap unless one range clears the other
        ov = "no" if (min(A) > max(B) or min(B) > max(A)) else "YES"
        print(f"  {name:<12} image {A.mean():8.2f} [{A.min():.2f},{A.max():.2f}]   "
              f"scalar {B.mean():8.2f} [{B.min():.2f},{B.max():.2f}]   "
              f"p={MW(A, B):.3f}  overlap={ov}")
    print()
