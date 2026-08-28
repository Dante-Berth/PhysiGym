"""Download per-run histories for the ``action_mode=full`` (uniform-action) sweeps.

Companion to ``download_wandb_project.py``, which handles the ``targeted``
sweeps.  Two differences, both load-bearing:

1. ``parse_name`` there strips ``_targeted_<timestamp>``; these runs end
   ``_full_<timestamp>``, so that regex never fires and mode detection falls
   through to a fuzzy substring fallback.  Here the observation mode is read
   from ``config`` (``model.observation_mode``), which is authoritative, and
   the name is used only for the seed and the algo.

2. These two projects are NOT returned by ``wandb.Api().projects()`` -- the
   same server-side invisibility recorded in RESEARCH_LOG T1.  They are only
   reachable by addressing ``entity/project`` directly.  Do not "verify" a
   project exists by listing; list-absence proves nothing.

Projects:
    ..._TRAIN_RECTANGLE_TEST_NETWORKFIELD_NEW_CHEMO   train=rect,  test=netfield
    ..._TRAIN_NETWORK_FIELD_TEST_RECTANGLE_full       train=netfield, test=rect

The KEYS map and the "never pass keys= to scan_history" rule are inherited
verbatim from download_wandb_project.py; see its comment for why.
"""

import argparse
import os
import re
import multiprocessing as mp

import pandas as pd
import wandb

from download_wandb_project import KEYS


def flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        kk = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, kk + "."))
        else:
            out[kk] = v
    return out


def process(args):
    entity, project, run_id, run_name, mode, action_mode, outdir = args
    api = wandb.Api(timeout=60)

    algo = "RANDOM" if re.search(r"(?i)random_baseline", run_name) else "SAC"
    seed = re.search(r"seed(\d+)", run_name)
    seed = int(seed.group(1)) if seed else -1

    # a random policy ignores its observation, so all baseline seeds share a folder
    folder = "random_baseline" if algo == "RANDOM" else mode
    subdir = os.path.join(outdir, folder)
    os.makedirs(subdir, exist_ok=True)
    out_path = os.path.join(subdir, f"{algo}_seed{seed}_{run_id}.csv")

    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        raw = pd.DataFrame(list(run.scan_history()))
        if raw.empty:
            print(f"!  {run_name}: no rows")
            return None
        present = {k: v for k, v in KEYS.items() if k in raw.columns}
        df = raw[list(present)].rename(columns=present).sort_values("step")
        for missing in set(KEYS.values()) - set(df.columns):
            df[missing] = pd.NA
        df = df[list(KEYS.values())]
        df.to_csv(out_path, index=False)
        n_q = int(df["q_bias"].notna().sum())
        print(f"ok {folder}/{os.path.basename(out_path)}  rows={len(df)} q_rows={n_q}")
        return dict(mode=mode, algo=algo, seed=seed, run_id=run_id,
                    action_mode=action_mode, state=run.state,
                    rows=len(df), q_rows=n_q, name=run_name)
    except Exception as e:
        print(f"XX {run_name} ({run_id}): {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="thomas-phd")
    ap.add_argument("--project", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(base, a.outdir)
    os.makedirs(outdir, exist_ok=True)

    wandb.login()
    runs = list(wandb.Api(timeout=60).runs(f"{a.entity}/{a.project}"))
    print(f"{a.project}: {len(runs)} runs -> {outdir}")

    tasks = []
    for r in runs:
        f = flat(r.config)
        mode = f.get("model.observation_mode") or f.get("env.observation_mode") or "unknown"
        action_mode = f.get("wrapper.action_mode") or f.get("model.action_mode") or "unknown"
        tasks.append((a.entity, a.project, r.id, r.name, mode, action_mode, outdir))

    with mp.get_context("spawn").Pool(a.workers) as pool:
        recs = [r for r in pool.map(process, tasks) if r]

    man = pd.DataFrame(recs).sort_values(["algo", "mode", "seed"])
    man.to_csv(os.path.join(outdir, "manifest.csv"), index=False)
    print(f"\nmanifest: {len(man)} runs, "
          f"{int((man.q_rows > 0).sum())} with Q-calibration rows")
    print(man.groupby(['action_mode', 'mode']).size().to_string())


if __name__ == "__main__":
    main()
