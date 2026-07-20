#!/usr/bin/env python3
"""
Stage 3 diagnosis: why do scalar-observation modes overfit the rectangle
train geometry while image-observation modes generalise to network-field (OOD)?

Empirical test, no model needed — uses only surviving per-episode traces:
  - data.csv           : per-step actions (action_x/y/radius normalised 0..1) + outcomes
  - ic_XXXXXX.csv      : the initial tumour cell positions for that episode

Hypothesis: image modes aim the drug AT the tumour (injection tracks the tumour
centroid) in BOTH train and test geometries; scalar modes learn a fixed
rectangle-appropriate aiming policy that does NOT re-target when the test
geometry (network-field) moves the tumour, so their injection<->tumour
alignment collapses on test.

Metric per episode: mean distance between the (normalised) injection centre and
the (normalised) tumour centroid, averaged over the steps where a dose is
actually applied. Lower = better aiming. We compare train vs test, image vs
scalar. The overfit signature is: scalar test alignment >> scalar train
alignment (aiming degrades OOD), while image stays flat.
"""
import glob
import os
import re
import sys
import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/PhysiCell_vroom_vroom/data")

# domain extent (from ic files / config): cells live roughly in [x_min,x_max]^2.
# ic coords are in microns on a centred grid; normalise to 0..1 to match action_x/y.
# infer bounds from the data rather than hard-coding.
MODES = {
    # label : (glob for run dirs, family)
    "I2  (img cells+subs)":      ("best_hyperparameters_SAC_img_mc_cells_substrates_w_cell*", "image"),
    "I2m (img cells+subs+m1m2)": ("best_hyperparameters_SAC_img_mc_cells_substrates_m1m2_w_cell*", "image"),
    "I1  (img cells)":           ("best_hyperparameters_SAC_img_mc_cells_w_cell*", "image"),
    "S3s (scalar cells+subs)":   ("best_hyperparameters_SAC_spatial_scalars_cells_substrates_w_cell*", "scalar"),
    "S3m (scalar cells+m1m2)":   ("best_hyperparameters_SAC_spatial_scalars_cells_m1m2_w_cell*", "scalar"),
    "S3sm(scalar cells+subs+m1m2)": ("best_hyperparameters_SAC_spatial_scalars_cells_substrates_m1m2_w_cell*", "scalar"),
}

DOSE_THRESH = 0.05  # only count steps where the agent actually injects


def _load_bounds():
    """Infer tumour-coordinate bounds from a sample of ic files."""
    ics = glob.glob(f"{DATA}/best_hyperparameters_SAC_*/env*/*/episodes/*/ic_*.csv")[:200]
    lo, hi = np.inf, -np.inf
    for f in ics:
        try:
            df = pd.read_csv(f)
            lo = min(lo, df.x.min(), df.y.min())
            hi = max(hi, df.x.max(), df.y.max())
        except Exception:
            pass
    return lo, hi


def _tumour_centroid_norm(ic_path, lo, hi):
    df = pd.read_csv(ic_path)
    t = df[df.type.astype(str).str.contains("tumor", case=False, na=False)]
    if len(t) == 0:
        return None
    cx = (t.x.mean() - lo) / (hi - lo)
    cy = (t.y.mean() - lo) / (hi - lo)
    return cx, cy


def _episode_alignment(run_dir, lo, hi):
    """Mean injection<->tumour-centroid distance over dosed steps, for one episode."""
    dcsv = os.path.join(run_dir, "data.csv")
    if not os.path.exists(dcsv):
        return None
    # find the ic file for this run (ic_<runnum>.csv alongside data.csv)
    m = re.search(r"run_0*(\d+)", os.path.basename(run_dir))
    ic = None
    if m:
        cand = os.path.join(run_dir, f"ic_{int(m.group(1)):06d}.csv")
        if os.path.exists(cand):
            ic = cand
    if ic is None:
        cands = glob.glob(os.path.join(run_dir, "ic_*.csv"))
        ic = cands[0] if cands else None
    if ic is None:
        return None
    cen = _tumour_centroid_norm(ic, lo, hi)
    if cen is None:
        return None
    cx, cy = cen
    df = pd.read_csv(dcsv)
    dosed = df[df.action_dose > DOSE_THRESH]
    if len(dosed) == 0:
        return None
    d = np.sqrt((dosed.action_x - cx) ** 2 + (dosed.action_y - cy) ** 2)
    return float(d.mean()), float(df.number_tumor.iloc[-1]), len(dosed)


def analyse():
    lo, hi = _load_bounds()
    print(f"# tumour-coord bounds inferred: [{lo:.1f}, {hi:.1f}] microns\n")
    rows = []
    for label, (pat, fam) in MODES.items():
        run_root = glob.glob(f"{DATA}/{pat}")
        for split in ("train", "test"):
            aligns, finals = [], []
            for rd in run_root:
                eps = glob.glob(f"{rd}/env*/{split}/episodes/run_*")
                # subsample episodes per run for speed but keep it representative
                for ep in eps[::5]:
                    r = _episode_alignment(ep, lo, hi)
                    if r is not None:
                        aligns.append(r[0])
                        finals.append(r[1])
            if aligns:
                rows.append(dict(
                    mode=label, family=fam, split=split,
                    n_eps=len(aligns),
                    align_mean=np.mean(aligns), align_sd=np.std(aligns),
                    final_tumor_med=np.median(finals),
                ))
    res = pd.DataFrame(rows)
    if res.empty:
        print("NO DATA — check paths"); return
    piv = res.pivot_table(index=["family", "mode"], columns="split",
                          values="align_mean")
    piv["test_minus_train"] = piv.get("test", np.nan) - piv.get("train", np.nan)
    pd.set_option("display.width", 140, "display.float_format", lambda v: f"{v:.4f}")
    print("=== injection<->tumour alignment (mean normalised distance; lower=better aiming) ===")
    print(piv.to_string())
    print("\n=== full table (with episode counts, final tumour, sd) ===")
    print(res.sort_values(["family", "mode", "split"]).to_string(index=False))
    res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "stage3_alignment.csv"), index=False)
    print("\nsaved -> stage3_alignment.csv")


if __name__ == "__main__":
    analyse()
