"""Measure the chemotaxis confound: the reverse targeted sweep, run twice.

Ch. 6 §6.1.1 lists two things that differ between the targeted and uniform
conditions (worker count -> batch size, and the action rate limit).  There is a
third, and it is not a training setting: the simulator changed.  Commit
f06d52a5 (2026-08-05 12:01 +0200 = 10:01 UTC) moved the advanced-chemotaxis
sensitivity onto the T cell, which thereafter climbs `anti_tumoral_factor` at
+0.3 and is repelled by `pro_tumoral_factor` at -0.15.  Every *targeted* run in
the ablation predates that commit; every *uniform* run postdates it.  So the
action space and the environment dynamics are perfectly confounded.

That confound can be MEASURED rather than merely disclosed, because the reverse
targeted sweep was run twice, once on each side of the change, with everything
else matched:

    old chemotaxis  ..._TRAIN_NETWORKFIELD_TEST_RECTANGLE        55 runs, 07-29..08-04
    new chemotaxis  ..._TRAIN_NETWORKFIELD_TEST_RECTANGLE_NEW_CHEMO  56 runs, 08-05..08-20

Both are action_mode=targeted, network_field -> rectangle, num_envs=28,
batch=1792, same nine observation modes, five seeds each.  The only deliberate
difference is the chemotaxis block.  The second sweep is used by nothing else in
this thesis; it is downloaded here purely as the control.

AGGREGATION IS IMPORTED, NOT REIMPLEMENTED -- same rule as
plot_action_mode_ablation: EWMA-50 per seed, <=5 distinct seeds by lowest
seed-id, crash filter MIN_ROWS=500, and the endpoint statistic of
`endpoints()` (last 50 smoothed train points, last 20 test).  So these numbers
are on the same scale as Tables 5.1/5.3 and as §6.1's.

Run from figures_plotting/.
"""
import glob
import os

import numpy as np
import pandas as pd

import plot_learning_curves_both_directions as base
from plot_learning_curves_both_directions import (
    BASE, MAX_SEEDS, MODE_META, MODE_ORDER, _ewma, _seed_id,
)
from plot_action_mode_ablation import MIN_ROWS

SWEEPS = {
    "old": "wandb_train_networkfield_test_rectangle",
    "new": "wandb_newchemo_train_networkfield_test_rectangle",
}
TAIL = {"train_return": 50, "test_return": 20}


def seed_files(sweep, mode):
    d = os.path.join(BASE, SWEEPS[sweep], mode)
    prefix = "RANDOM_" if mode == "random_baseline" else "SAC_"
    files = sorted(glob.glob(os.path.join(d, prefix + "*.csv")), key=_seed_id)
    files = [f for f in files if sum(1 for _ in open(f)) - 1 >= MIN_ROWS]
    seen, dedup = set(), []
    for f in files:
        s = _seed_id(f)
        if s not in seen:
            seen.add(s)
            dedup.append(f)
    return dedup[:MAX_SEEDS]


def endpoint(f, col):
    d = pd.read_csv(f, usecols=["step", col]).dropna().sort_values("step")
    if len(d) <= 3:
        return None
    return _ewma(d[col].to_numpy())[-TAIL[col]:].mean()


def per_seed(sweep, mode, col):
    """seed -> endpoint, so the two sweeps can be matched seed by seed."""
    out = {}
    for f in seed_files(sweep, mode):
        v = endpoint(f, col)
        if v is not None:
            out[_seed_id(f)] = v
    return out


def main():
    rows = []
    for col in ("train_return", "test_return"):
        for mode in MODE_ORDER:
            if mode == "random_baseline":
                continue
            o, n = per_seed("old", mode, col), per_seed("new", mode, col)
            shared = sorted(set(o) & set(n))
            rows.append(dict(
                split=col.split("_")[0], id=MODE_META[mode]["id"], mode=mode,
                n_old=len(o), n_new=len(n), n_paired=len(shared),
                old_mu=np.mean(list(o.values())) if o else np.nan,
                new_mu=np.mean(list(n.values())) if n else np.nan,
                paired_delta=(np.mean([n[s] - o[s] for s in shared])
                              if shared else np.nan),
            ))
    df = pd.DataFrame(rows)
    df["delta"] = df["new_mu"] - df["old_mu"]
    out = os.path.join(BASE, "out_action_mode_ablation", "chemotaxis_control.csv")
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    for split in ("train", "test"):
        d = df[df.split == split]
        print(f"\n=== {split} return, network-field -> rectangle, targeted, n_envs=28")
        print(d[["id", "n_old", "n_new", "n_paired", "old_mu", "new_mu",
                 "delta", "paired_delta"]].to_string(index=False,
                 float_format=lambda v: f"{v:8.2f}"))
        v = d["delta"].dropna().to_numpy()
        pv = d["paired_delta"].dropna().to_numpy()
        print(f"  across modes: mean delta {v.mean():+.2f}  "
              f"(min {v.min():+.2f}, max {v.max():+.2f}, n={len(v)})")
        if len(pv):
            print(f"  seed-matched: mean delta {pv.mean():+.2f}  "
                  f"(min {pv.min():+.2f}, max {pv.max():+.2f}, n={len(pv)})")
            print(f"  sign         : {int((pv > 0).sum())}/{len(pv)} modes higher "
                  f"under new chemotaxis")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
