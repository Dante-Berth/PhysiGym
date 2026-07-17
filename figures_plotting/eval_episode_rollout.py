"""Deterministic eval rollouts from trained SAC checkpoints, for the paper's
episode-comparison figure. Runs I2 / S3s / POMDP from IDENTICAL network-field
initial conditions and logs per-step tumor count, dose, and reward to CSV.

Everything is checkpoint-driven: each .pt carries its own `config` (d_arg) and
`d_arg_env`, so we rebuild the exact env + actor from the checkpoint.

Run from the vroom repo root (needs config/ and the extending.physicell .so):
    cd /home/alex/PhysiCell_vroom_vroom
    ENV=custom_modules/physigym/physigym/envs
    PYTHONPATH=$ENV .venv/bin/python \
        /home/alex/PhysiGym/figures_plotting/eval_episode_rollout.py \
        --mode POMDP --ic_seed 1000 --out /home/alex/PhysiGym/figures_plotting/episode_rollouts
"""
import os, sys, argparse, csv
import numpy as np
import torch

import glob as _glob

CKPT_DIR = {
    "I2":    "data/best_hyperparameters_SAC_img_mc_cells_substrates_w_cell=0.3_w_dose=2.0_w_smooth=0.0_seed42_42_img_mc_cells_substrates_targeted_1784165317/checkpoints",
    "S3s":   "data/best_hyperparameters_SAC_spatial_scalars_cells_substrates_w_cell=0.3_w_dose=2.0_w_smooth=0.0_seed1_1_spatial_scalars_cells_substrates_targeted_1783734211/checkpoints",
    "POMDP": "data/best_hyperparameters_SAC_scalars_macrophages_w_cell=0.3_w_dose=2.0_w_smooth=0.0_seed42_42_scalars_macrophages_targeted_1784203370/checkpoints",
}


def _resolve_ckpt(mode):
    d = CKPT_DIR[mode]
    for name in ("sac_final.pt", "sac_latest.pt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    steps = sorted(_glob.glob(os.path.join(d, "sac_step*.pt")))
    if not steps:
        raise FileNotFoundError(f"no checkpoint in {d}")
    return steps[-1]


CKPT = {m: _resolve_ckpt(m) for m in CKPT_DIR}


def build(mode, ic_seed, replay_ic=None):
    from vectorized import make_physigym_env
    from nn import Actor

    ckpt = torch.load(CKPT[mode], map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # force: single env, all-test network-field episodes, no wandb, minimal sim
    cfg["vectorization"]["num_envs"] = 1
    cfg["vectorization"]["rl_threads"] = 1
    # single-threaded: PhysiCell's mechanics/motility loops draw from a shared
    # UniformRandom() stream inside OpenMP-parallel loops, so with >1 thread the
    # draw order (and therefore the whole trajectory) is scheduler-dependent and
    # NOT reproducible run-to-run even with a fixed seed + fixed IC.
    cfg["vectorization"]["threads_per_env"] = 1
    cfg["simulation"]["wandb_track"] = False
    cfg["model"]["render_mode"] = None
    cfg["generation"]["mode_test"] = ["network_field"]
    cfg["generation"]["mode_train"] = ["network_field"]
    cfg["generation"]["seed"] = ic_seed
    cfg["wrapper"]["frequence_episode_test"] = 1   # every episode is a test episode
    tag = f"{mode}_{ic_seed}" if replay_ic is None else f"{mode}_replay"
    cfg["model"]["output_dir"] = os.path.join("data", "_eval_tmp", tag)

    # replay mode: force an identical initial condition across modes.
    # NOTE: do NOT reset() here — init_fn() already performs the env's first
    # reset internally, and rollout() below does the real replay reset right
    # before stepping. An extra reset() here previously double-reset the env,
    # which perturbed PhysiCell's shared RNG stream and made rollouts
    # non-reproducible even with a fixed seed + fixed IC.
    init_fn = make_physigym_env(0, cfg)
    env = init_fn()

    actor = Actor(ckpt["d_arg_env"], cfg["neural_architecture_image"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return env, actor, cfg


def rollout(mode, ic_seed, out_dir, max_steps=672, replay_ic=None):
    env, actor, cfg = build(mode, ic_seed, replay_ic=replay_ic)
    os.makedirs(out_dir, exist_ok=True)
    if replay_ic is not None:
        obs, info = env.reset(no_generation_cfg={"list_csv": [os.path.abspath(replay_ic)],
                                                 "dataset": "replay"})
    else:
        obs, info = env.reset()
    rows, cum_r, cum_dose = [], 0.0, 0.0
    for t in range(max_steps):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            # deterministic: use the tanh-mean action, not a sample
            a, _, mean = actor.get_action(x)
            act = mean.squeeze(0).cpu().numpy()
        obs, r, term, trunc, info = env.step(act)
        cum_r += float(r)
        cum_dose += float(info.get("dose_spent", 0.0))
        rows.append(dict(step=t, mode=mode, ic_seed=ic_seed,
                         reward=float(r), cum_reward=cum_r,
                         dose=float(info.get("dose_spent", 0.0)), cum_dose=cum_dose,
                         number_tumor=int(info.get("number_tumor", -1)),
                         action_dose=float(act[0]),
                         action_x=float(act[1]) if len(act) > 1 else 0.0,
                         action_y=float(act[2]) if len(act) > 2 else 0.0,
                         action_radius=float(act[3]) if len(act) > 3 else 0.0))
        if term or trunc:
            break
    fn = os.path.join(out_dir, f"{mode}_seed{ic_seed}.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[{mode} seed{ic_seed}] steps={len(rows)} final_tumor={rows[-1]['number_tumor']} "
          f"cum_reward={cum_r:.1f} cum_dose={cum_dose:.1f} -> {fn}")
    env.close()
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(CKPT), required=True)
    ap.add_argument("--ic_seed", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=672)
    ap.add_argument("--out", default="/home/alex/PhysiGym/figures_plotting/episode_rollouts")
    ap.add_argument("--replay_ic", default=None, help="path to a fixed IC csv to replay")
    a = ap.parse_args()
    rollout(a.mode, a.ic_seed, a.out, a.max_steps, replay_ic=a.replay_ic)
