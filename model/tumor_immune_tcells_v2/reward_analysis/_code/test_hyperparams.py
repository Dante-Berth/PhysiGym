"""
Quick sanity test for publication hyperparameters.
Tests two policies per episode:
  - random:   uniform random actions
  - full_dose: action_mode=full, dose=1.0 always (max drug, centered)

Checks:
  - delta constraints are respected (max delta <= limit)
  - reward signal is non-trivial
  - cumulative return grows/changes over the episode

Run from:
  cd /home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs
  python test_hyperparams.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/alex/Physi/PhysiCell/custom_modules/extending/build/lib.linux-x86_64-cpython-311")

import numpy as np
from vectorized_tip import test_make_physigym_env

# ── Hyperparameters to validate ──────────────────────────────
ACTION_REPEAT  = 4
DELTA_X        = 0.15
DELTA_Y        = 0.15
DELTA_RADIUS   = 0.20
DELTA_DOSE     = 1.0   # unconstrained
W_SMOOTH       = 0.02
N_EPISODES     = 3     # episodes per policy (short run)
MAX_STEPS_EP   = 200   # cap steps per episode to keep runtime short

params = {
    "tumor":      {"correlation_length": 45, "threshold": 0.55, "number_cells": 64},
    "t_cell":     {"correlation_length": 45, "threshold": 0.55, "number_cells": 8},
    "macrophage": {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
}

BASE_CFG = {
    "simulation": {"max_time": 1440.0, "seed": 42},   # 1 day = faster episodes
    "vectorization": {"num_envs": 1, "rl_threads": 2},
    "model": {
        "id": "physigym/ModelPhysiCellEnv-v0",
        "settingxml":   "config/PhysiCell_settings.xml",
        "settingcells": "config/cells.csv",
        "output_dir":   "./test_hyperparam_output",
        "figsize": (6, 6),
        "observation_mode": "scalars_macrophages",
        "render_mode": None,
        "verbose": False,
        "img_rgb_grid_size_x": 64,
        "img_rgb_grid_size_y": 64,
        "img_mc_grid_size_x":  64,
        "img_mc_grid_size_y":  64,
        "normalization_factor": 512,
        "action_mode": "targeted",
    },
    "wrapper": {
        "list_variable_name": ["drug_1_dose", "drug_1_x", "drug_1_y", "drug_1_radius"],
        "w_cell":          0.3,
        "w_smooth":        W_SMOOTH,
        "action_delta_max": [DELTA_DOSE, DELTA_X, DELTA_Y, DELTA_RADIUS],
        "action_mode":     "targeted",
        "frequence_episode_test": 4,
    },
    "generation": {
        "params":     params,
        "seed":       42,
        "mode_train": ["rectangle"],
        "mode_test":  ["rectangle"],
    },
    "rl": {"total_timesteps": 10_000},
}


def run_policy(env, policy_fn, n_episodes, max_steps, action_repeat, label, is_targeted=True):
    """
    Runs policy_fn for n_episodes.
    Delta constraint is checked per decision-step (one raw action = action_repeat env steps).
    Tracking compares the first executed action of each decision block to the previous block's first.
    """
    print(f"\n{'='*60}")
    print(f"  Policy: {label}")
    print(f"{'='*60}")

    all_ep_returns = []

    for ep in range(n_episodes):
        env.reset()
        ep_reward   = 0.0
        ep_dose     = 0.0
        cumulative  = []
        # track executed action at the START of each decision block
        prev_block_action = None
        deltas_x, deltas_y, deltas_r = [], [], []
        step = 0
        done = False

        while not done and step < max_steps:
            action = policy_fn(env)

            # snapshot history length before this block
            hist_len_before = len(env._action_history)

            acc_reward = 0.0
            last_info  = {}
            for _rep in range(action_repeat):
                obs, reward, terminated, truncated, info = env.step(action)
                acc_reward += reward
                last_info   = info
                step       += 1
                if terminated or truncated:
                    done = True
                    break

            ep_reward += acc_reward
            ep_dose   += float(last_info.get("dose_spent", 0.0))
            cumulative.append(ep_reward)

            # first executed action in this block (after delta-clip on sub-step 0)
            if len(env._action_history) > hist_len_before and is_targeted:
                block_first = env._action_history[hist_len_before]
                if prev_block_action is not None:
                    deltas_x.append(abs(float(block_first[1]) - float(prev_block_action[1])))
                    deltas_y.append(abs(float(block_first[2]) - float(prev_block_action[2])))
                    deltas_r.append(abs(float(block_first[3]) - float(prev_block_action[3])))
                prev_block_action = block_first.copy()

        # ── Constraint verification ──────────────────────────
        max_dx = max(deltas_x) if deltas_x else 0.0
        max_dy = max(deltas_y) if deltas_y else 0.0
        max_dr = max(deltas_r) if deltas_r else 0.0

        ok_x = max_dx <= DELTA_X + 1e-4
        ok_y = max_dy <= DELTA_Y + 1e-4
        ok_r = max_dr <= DELTA_RADIUS + 1e-4

        tumor_end = last_info.get("number_tumor", "?")
        autocorr  = last_info.get("action_autocorr_lag1", float("nan"))

        print(f"\n  Episode {ep+1}/{n_episodes}")
        print(f"    decision_steps={step//action_repeat}  env_steps={step}  "
              f"cumulative_return={ep_reward:+.3f}  dose_total={ep_dose:.3f}  tumor_end={tumor_end}")
        print(f"    autocorr_lag1 = {autocorr:.3f}  (higher = smoother)")
        if is_targeted:
            print(f"    Δx  max={max_dx:.4f}  limit={DELTA_X}  "
                  f"{'✓ OK' if ok_x else '✗ VIOLATED'}")
            print(f"    Δy  max={max_dy:.4f}  limit={DELTA_Y}  "
                  f"{'✓ OK' if ok_y else '✗ VIOLATED'}")
            print(f"    Δr  max={max_dr:.4f}  limit={DELTA_RADIUS}  "
                  f"{'✓ OK' if ok_r else '✗ VIOLATED'}")

        n = len(cumulative)
        if n >= 5:
            idx = [0, n//4, n//2, 3*n//4, n-1]
            pts = [f"t={i}: {cumulative[i]:+.2f}" for i in idx]
            print(f"    cumulative curve: {' | '.join(pts)}")

        all_ep_returns.append(ep_reward)

    print(f"\n  [{label}] mean return over {n_episodes} eps: "
          f"{np.mean(all_ep_returns):+.3f} ± {np.std(all_ep_returns):.3f}")
    return all_ep_returns


def main():
    print(f"\n[HYPERPARAM VALIDATION]")
    print(f"  action_repeat={ACTION_REPEAT}  w_smooth={W_SMOOTH}")
    print(f"  delta [dose={DELTA_DOSE}, x={DELTA_X}, y={DELTA_Y}, r={DELTA_RADIUS}]")

    import copy
    cfg = copy.deepcopy(BASE_CFG)

    # Only one PhysiCell instance can exist per process — run both policies
    # sequentially on the SAME env, switching wrapper config between episodes.
    env = test_make_physigym_env(cfg)

    # ── Policy 1: random targeted (delta-clipped) ────────────
    random_returns = run_policy(
        env,
        policy_fn=lambda e: np.random.rand(4).astype(np.float32),
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS_EP,
        action_repeat=ACTION_REPEAT,
        label="random (targeted, delta-clipped)",
        is_targeted=True,
    )

    # ── Switch to full-dose mode by patching wrapper attrs ───
    print("\n[switching env to action_mode=full, dose=1.0 always]")
    env.action_mode        = "full"
    env.list_variable_name = ["drug_1_dose"]
    env.w_smooth           = 0.0
    env._action_delta_max  = None

    full_dose_returns = run_policy(
        env,
        policy_fn=lambda e: np.array([1.0], dtype=np.float32),
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS_EP,
        action_repeat=ACTION_REPEAT,
        label="full dose=1.0 (centered, max radius)",
        is_targeted=False,
    )

    env.close()

    # ── Summary comparison ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  random targeted:  {np.mean(random_returns):+.3f} ± {np.std(random_returns):.3f}")
    print(f"  full dose=1.0:    {np.mean(full_dose_returns):+.3f} ± {np.std(full_dose_returns):.3f}")
    diff = np.mean(full_dose_returns) - np.mean(random_returns)
    print(f"  Δ (full - random): {diff:+.3f}  "
          f"({'full dose kills more tumor relative to cost' if diff > 0 else 'random wins or reward scale issue'})")


if __name__ == "__main__":
    os.chdir("/home/alex/Physi/PhysiCell")
    main()
