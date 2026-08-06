"""Download full per-run histories for one wandb project into its own folder.

Supersedes ``download_all_wandb_histories.py``, which hard-filtered
``scan_history(keys=["_step", return, length])`` and therefore silently dropped
the Q-calibration metrics (``charts/test_q_bias``, ``charts/test_q_mae``) that
``run.py`` does log.  Those metrics are the third line of evidence for Ch. 6.

One project per output directory, on purpose: the two transfer directions live
in two different wandb projects and must never be merged into one folder.

    SAC_ASYNC_TME_NEW_HYP_REWARD
        train = rectangle,     test = network-field   (the EWRL-reported sweep)
    SAC_ASYNC_TME_NEW_HYP_REWARD_TRAIN_NETWORKFIELD_TEST_RECTANGLE
        train = network-field, test = rectangle       (reverse transfer, T2)

Layout written (compatible with plot_tme_new.py, plus the extra columns):

    <outdir>/manifest.csv
    <outdir>/<mode>/SAC_seed<S>_<runid>.csv
    <outdir>/random_baseline/RANDOM_seed<S>_<runid>.csv

Usage:
    python download_wandb_project.py \
        --project SAC_ASYNC_TME_NEW_HYP_REWARD \
        --outdir  wandb_train_rectangle_test_networkfield
"""

import argparse
import os
import re
import multiprocessing as mp

import pandas as pd
import wandb

# ── the metrics we keep, and the short column names they land under ──────────
# ⚠️ Do NOT pass these to scan_history(keys=...).  That form intersects: it
# returns only rows where *every* requested key is present, and since returns
# are logged per episode while the Q/loss metrics tick every 500 gradient
# steps, asking for both at once yields a handful of coincidental rows (2-14
# per run instead of ~1700).  Scan the full history and select afterwards.
KEYS = {
    "_step": "step",
    "charts/train_return_mean": "train_return",
    "charts/train_return_std": "train_std",
    "charts/test_return_mean": "test_return",
    "charts/test_return_std": "test_std",
    "charts/train_return_raw": "train_return_raw",
    "charts/test_return_raw": "test_return_raw",
    "charts/train_episode_length": "train_length",
    "charts/test_episode_length": "test_length",
    # value calibration — the point of this rewrite
    "charts/test_q_bias": "q_bias",  # mean(Q - MC), signed; > 0 = overestimation
    "charts/test_q_mae": "q_mae",  # magnitude of miscalibration
    "charts/test_q_corr": "q_corr",  # kept for completeness; see RESEARCH_LOG Q2
    "charts/qf1_loss": "qf1_loss",
    "charts/qf2_loss": "qf2_loss",
    "charts/alpha": "alpha",
    # behavioural, split-wise
    "charts/train_action_delta_mean": "train_action_delta_mean",
    "charts/test_action_delta_mean": "test_action_delta_mean",
    "charts/train_action_delta_std": "train_action_delta_std",
    "charts/test_action_delta_std": "test_action_delta_std",
    "charts/train_action_autocorr_lag1": "train_action_autocorr",
    "charts/test_action_autocorr_lag1": "test_action_autocorr",
    "samples_drained": "samples_drained",
}

MODES = [
    "img_mc_cells_substrates_m1m2",
    "img_mc_cells_substrates",
    "img_mc_cells_m1m2",
    "img_mc_cells",
    "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2",
    "spatial_scalars_cells_substrates_m1m2",
    "spatial_scalars_cells_substrates",
    "spatial_scalars_cells_m1m2",
    "scalars_macrophages",
]


def parse_name(name):
    """(algo, mode, seed) from a run name.

    The observation mode is the token block immediately preceding
    ``_targeted_<timestamp>``; matched against MODES longest-first so that
    e.g. ``img_mc_cells_substrates_m1m2`` is not truncated to ``img_mc_cells``.

    ⚠️ The two projects name their random-policy baselines differently --
    ``best_hyperparameters_RANDOM_baseline_...`` in the rectangle->network-field
    sweep, ``random_baseline_...`` in the reverse one.  Matching only the second
    files every forward baseline as if it were a SAC run, where its low seed-id
    then displaces a real SAC seed from the first-5-by-seed-id selection.
    """
    algo = "RANDOM" if re.search(r"(?i)random_baseline", name) else "SAC"

    seed = re.search(r"seed(\d+)", name)
    seed = int(seed.group(1)) if seed else -1

    tail = re.sub(r"_targeted_\d+$", "", name)
    mode = next((m for m in sorted(MODES, key=len, reverse=True)
                 if tail.endswith(m)), None)
    if mode is None:  # fall back to whichever mode appears anywhere in the name
        mode = next((m for m in sorted(MODES, key=len, reverse=True)
                     if m in name), "unknown")
    return algo, mode, seed


def process(args):
    entity, project, run_id, run_name, outdir = args
    api = wandb.Api(timeout=60)  # MUST be created inside the process

    algo, mode, seed = parse_name(run_name)
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
    runs = wandb.Api(timeout=60).runs(f"{a.entity}/{a.project}")
    print(f"{a.project}: {len(runs)} runs -> {outdir}")

    tasks = [(a.entity, a.project, r.id, r.name, outdir) for r in runs]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        recs = [r for r in pool.map(process, tasks) if r]

    man = pd.DataFrame(recs).sort_values(["algo", "mode", "seed"])
    man.to_csv(os.path.join(outdir, "manifest.csv"), index=False)
    print(f"\nmanifest: {len(man)} runs, "
          f"{int((man.q_rows > 0).sum())} with Q-calibration rows")


if __name__ == "__main__":
    main()
