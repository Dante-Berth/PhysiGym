#!/usr/bin/env python3
"""C6: drug-aiming on the REVERSE transfer direction (train network-field, test rectangle).

Same metric and conventions as stage3_transfer.py, which measured the forward
direction: mean distance between the normalised injection centre and that
episode's own tumour centroid, over the steps where a dose is actually applied.

Two differences from the forward script, both required by C6:
  1. DATA is passed in, because the reverse traces are split across two hosts
     (image modes on sureli11, spatial-scalar modes on sureli9).
  2. The run-dir glob is SAC_<mode>_..._mode_train_network_field_..., not
     best_hyperparameters_SAC_<mode>_..., which matches only the forward sweep.

EXCLUSIONS, deliberate:
  - `new_chemotaxis` runs are dropped. The chemotaxis edit landed 2026-08-05,
    after both reported sweeps, so including them would compare across a model
    change (ch6_results.tex: "The two directions are therefore a matched pair").
  - `_full` (uniform-action) runs are dropped: aiming is undefined when the
    injection covers the whole domain.
"""
import glob, os, re, sys, json
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/PhysiCell_vroom_vroom/data")
DOSE_THRESH = 0.05          # same as the forward analysis
STRIDE = 5                  # same subsampling as the forward analysis (eps[::5])

MODES = {
    "img_mc_cells": "image",
    "img_mc_cells_m1m2": "image",
    "img_mc_cells_substrates": "image",
    "img_mc_cells_substrates_m1m2": "image",
    "spatial_scalars_cells_substrates": "scalar",
    "spatial_scalars_cells_m1m2": "scalar",
    "spatial_scalars_cells_substrates_m1m2": "scalar",
    "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2": "scalar",
}


def run_dirs(mode):
    """Reverse-direction, targeted-action, non-chemotaxis run dirs for one mode."""
    out = []
    for d in glob.glob(f"{DATA}/SAC_{mode}_w_cell*mode_train_network_field*"):
        b = os.path.basename(d)
        # the mode must match exactly: img_mc_cells must not swallow img_mc_cells_m1m2
        if not re.match(rf"^SAC_{re.escape(mode)}_w_cell=", b):
            continue
        if "new_chemotaxis" in b or b.endswith("_full") or "_full_" in b:
            continue
        out.append(d)
    return sorted(out)


def _bounds(sample):
    lo, hi = np.inf, -np.inf
    for f in sample:
        try:
            df = pd.read_csv(f, usecols=["x", "y"])
            lo = min(lo, df.x.min(), df.y.min())
            hi = max(hi, df.x.max(), df.y.max())
        except Exception:
            pass
    return float(lo), float(hi)


def _centroid(ic_path, lo, hi):
    df = pd.read_csv(ic_path)
    t = df[df.type.astype(str).str.contains("tumor", case=False, na=False)]
    if len(t) == 0:
        return None
    return ((t.x.mean() - lo) / (hi - lo), (t.y.mean() - lo) / (hi - lo))


def _episode(run_dir, lo, hi):
    dcsv = os.path.join(run_dir, "data.csv")
    if not os.path.exists(dcsv):
        return None
    m = re.search(r"run_0*(\d+)", os.path.basename(run_dir))
    ic = os.path.join(run_dir, f"ic_{int(m.group(1)):06d}.csv") if m else None
    if not (ic and os.path.exists(ic)):
        c = glob.glob(os.path.join(run_dir, "ic_*.csv"))
        ic = c[0] if c else None
    if ic is None:
        return None
    cen = _centroid(ic, lo, hi)
    if cen is None:
        return None
    cx, cy = cen
    try:
        df = pd.read_csv(dcsv)
    except Exception:
        return None
    if "action_x" not in df.columns:
        return None
    dosed = df[df.action_dose > DOSE_THRESH]
    if len(dosed) == 0:
        return None
    d = np.sqrt((dosed.action_x - cx) ** 2 + (dosed.action_y - cy) ** 2)
    return float(d.mean()), float(df.number_tumor.iloc[-1])


def main():
    ics = []
    for mode in MODES:
        for rd in run_dirs(mode)[:1]:
            ics += glob.glob(f"{rd}/env*/*/episodes/*/ic_*.csv")[:60]
    if not ics:
        print(json.dumps({"error": "no ic files", "data": DATA})); return
    lo, hi = _bounds(ics[:200])

    rows = []
    for mode, fam in MODES.items():
        rds = run_dirs(mode)
        if not rds:
            continue
        # NOTE: in the reverse direction, train = network-field and test = rectangle
        for split in ("train", "test"):
            al, fin, nseed = [], [], 0
            for rd in rds:
                eps = sorted(glob.glob(f"{rd}/env*/{split}/episodes/run_*"))
                if eps:
                    nseed += 1
                for ep in eps[::STRIDE]:
                    r = _episode(ep, lo, hi)
                    if r is not None:
                        al.append(r[0]); fin.append(r[1])
            if al:
                rows.append(dict(mode=mode, family=fam, split=split, n_runs=nseed,
                                 n_eps=len(al), align_mean=float(np.mean(al)),
                                 align_sd=float(np.std(al)),
                                 final_tumor_med=float(np.median(fin))))
    print(json.dumps({"bounds": [lo, hi], "rows": rows}))


if __name__ == "__main__":
    main()
