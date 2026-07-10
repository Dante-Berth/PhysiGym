import ctypes
import ctypes.util
import os

gomp_path = ctypes.util.find_library("gomp")
if gomp_path:
    ctypes.CDLL(gomp_path, mode=ctypes.RTLD_GLOBAL)

# Silence PhysiCell's C++ console output: the compiled physicell module mutes
# std::cout while PHYSIGYM_QUIET is set (read via std::getenv on each call).
# Set it here, before any env is created, so every path — the subprocess envs
# and the test harness — inherits it. Override by exporting PHYSIGYM_QUIET=0.
os.environ.setdefault("PHYSIGYM_QUIET", "1")
# os.environ["OMP_NUM_THREADS"] = "1"
# Maybe delete this because sometimes it is slowdown
# NOW import everything else
import shutil
import argparse

import os
import shutil
import argparse
import psutil
import gymnasium as gym
import numpy as np
from lxml import etree
import time
from tqdm import tqdm
import multiprocessing as mp

from resilient_sub_vec_env import ResilientSubprocVecEnv
from custom_modules.physigym.physigym.envs.wrapper import PhysiCellModelWrapper
import sys
import physigym
from extending import physicell


# ============================================================
# Helper: Unique Seed Logic
# ============================================================
def get_unique_seed(master_seed, env_id, total_envs):
    """
    Generates a statistically independent seed for each worker.
    """
    ss = np.random.SeedSequence(master_seed)
    # Spawn enough seeds for all envs + 1 for the test environment
    child_seeds = ss.spawn(total_envs + 1)
    return int(child_seeds[env_id].generate_state(1)[0] % (2**31))


# ============================================================
# Helper: CPU affinity
# ============================================================
def assign_cpu_affinity(env_id: int, threads_per_env: int, offset_threads: int):
    os.environ["OMP_NUM_THREADS"] = str(threads_per_env)
    total_cores = psutil.cpu_count(logical=True)
    start = env_id * threads_per_env + offset_threads
    end = min(start + threads_per_env, total_cores)
    core_list = list(range(start, end))
    try:
        psutil.Process().cpu_affinity(core_list)
    except Exception:
        pass  # Handle systems where affinity isn't supported


# ============================================================
# Helper: Environment factory
# ============================================================
def make_physigym_env(env_id: int, cfg: dict):
    sim_cfg = cfg["simulation"]
    vect_cfg = cfg["vectorization"]
    model_cfg = cfg["model"]
    wrapper_cfg = cfg["wrapper"]
    generation_cfg = cfg["generation"].copy()

    base_xml = model_cfg["settingxml"]
    base_cells = model_cfg["settingcells"]
    master_seed = sim_cfg["seed"] if sim_cfg["seed"] is not None else 42

    os.makedirs("config", exist_ok=True)
    env_xml = f"config/PhysiCell_settings_env{env_id}.xml"
    env_cells = f"config/cells_{env_id}.csv"

    def _init():
        # ===== CRITICAL: Preload OpenMP in subprocess =====
        import ctypes
        import ctypes.util

        gomp_path = ctypes.util.find_library("gomp")
        if gomp_path:
            ctypes.CDLL(gomp_path, mode=ctypes.RTLD_GLOBAL)
        # ==================================================
        rl_threads = vect_cfg["rl_threads"]
        threads_per_env = vect_cfg["threads_per_env"]
        assign_cpu_affinity(env_id, threads_per_env, offset_threads=rl_threads)

        # Unique seed for this process
        worker_seed = get_unique_seed(master_seed, env_id, vect_cfg["num_envs"])

        # Prepare files
        shutil.copy(base_xml, env_xml)
        shutil.copy(base_cells, env_cells)

        # Modify XML
        tree = etree.parse(env_xml)
        root = tree.getroot()
        root.xpath("//overall/max_time")[0].text = str(sim_cfg["max_time"])
        root.xpath("//parallel/omp_num_threads")[0].text = str(threads_per_env)

        out_path = os.path.join(model_cfg.get("output_dir", "output"), f"env{env_id}")
        os.makedirs(out_path, exist_ok=True)
        root.xpath("//save/folder")[0].text = out_path
        root.xpath("//initial_conditions/cell_positions/filename")[0].text = env_cells
        tree.write(env_xml, pretty_print=True)

        local_model_cfg = model_cfg.copy()
        local_model_cfg["settingxml"] = env_xml
        if "settingcells" in local_model_cfg:
            del local_model_cfg["settingcells"]
        if "output_dir" in local_model_cfg:
            del local_model_cfg["output_dir"]

        env = gym.make(**local_model_cfg)
        env = PhysiCellModelWrapper(env, **wrapper_cfg)

        # Create a copy of generation_cfg to avoid modifying the shared dict
        gen_cfg_copy = generation_cfg.copy()
        gen_cfg_copy["seed"] = worker_seed
        env.reset(generation_cfg=gen_cfg_copy)
        return env

    return _init


# ============================================================
# Single Env Test Function (Restored & Fixed)
# ============================================================
def test_make_physigym_env(cfg: dict, env_id=0):
    """
    Creates a single instance for testing/debugging.
    """
    print(f"[TEST] Initializing test environment {env_id}...")

    # Calculate threads like the vectorizer would
    vect_cfg = cfg["vectorization"]
    rl_threads = vect_cfg["rl_threads"]
    total_cores = psutil.cpu_count(logical=True)
    threads_per_env = (total_cores - rl_threads) // vect_cfg["num_envs"]
    cfg["vectorization"]["threads_per_env"] = max(1, threads_per_env)

    # Use the factory logic directly to ensure consistency
    init_fn = make_physigym_env(env_id, cfg)
    return init_fn()


def _build_env_fns(cfg: dict):
    """Compute the thread layout and build one factory per env (shared by the
    batched and async vec-env constructors)."""
    vect_cfg = cfg["vectorization"]
    num_envs = vect_cfg["num_envs"]
    rl_threads = vect_cfg["rl_threads"]

    total_cores = psutil.cpu_count(logical=True)
    threads_per_env = (total_cores - rl_threads) // num_envs
    cfg["vectorization"]["threads_per_env"] = max(1, threads_per_env)

    print(f"[INFO] Launching {num_envs} envs × {threads_per_env} threads each")
    return [make_physigym_env(i, cfg) for i in range(num_envs)]


def vec_envs(cfg: dict):
    """Batched (lock-step) vec env — the original behavior."""
    env_fns = _build_env_fns(cfg)
    return ResilientSubprocVecEnv(env_fns=env_fns, start_method="spawn")


def vec_envs_async(cfg: dict):
    """Asynchronous per-env vec env: exposes step_send/poll_ready/recv_env so
    the actor loop can re-dispatch each env the instant it returns, instead of
    blocking on the slowest env every step. Falls back to the batched class if
    the async module is unavailable."""
    from resilient_async_vec_env import ResilientAsyncSubprocVecEnv

    env_fns = _build_env_fns(cfg)
    return ResilientAsyncSubprocVecEnv(env_fns=env_fns, start_method="spawn")


# ============================================================
# Runner
# ============================================================
def run_vectorized(cfg: dict):
    envs = vec_envs(cfg)
    envs.reset()

    num_envs = envs.num_envs
    total = cfg["rl"]["total_timesteps"]
    pbar = tqdm(total=total)
    local_step = 0

    while local_step < total:
        actions = np.array([[0.0] for _ in range(num_envs)], dtype=np.float32)
        _, _, _, _ = envs.step(actions)
        local_step += num_envs
        pbar.update(num_envs)

    envs.close()
    pbar.close()


import matplotlib.pyplot as plt
import math
import numpy as np


def plot_and_save_observation(env, obs, save_path="observation.png"):
    """
    Takes an image-based observation from ModelPhysiCellEnv and plots
    each channel as a separate heatmap, then saves it to disk safely.
    """
    # 1. Dynamically extract channel names using get_wrapper_attr to bypass the Wrapper
    cell_dict = env.get_wrapper_attr("cell_type_to_id")
    substrates = env.get_wrapper_attr("substrate_unique")

    # Sort cell types by their assigned ID to match get_matrix_cells()
    sorted_cells = sorted(cell_dict.items(), key=lambda item: item[1])
    cell_names = [f"Cell: {cell[0]}" for cell in sorted_cells]

    # Substrates follow the cells
    substrate_names = [f"Subs: {sub}" for sub in substrates]

    channel_names = cell_names + substrate_names
    num_channels = obs.shape[0]

    # 2. Setup the subplot grid dynamically based on channel count
    ncols = 3
    nrows = math.ceil(num_channels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))

    # Flatten axes array for easy iteration (handles 1D and 2D axes arrays)
    if num_channels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 3. Plot each channel
    for i in range(num_channels):
        ax = axes[i]

        # obs[i] is (X_bins, Y_bins). We transpose (.T) so X is horizontal and Y is vertical.
        im = ax.imshow(obs[i].T, cmap="turbo", origin="lower", vmin=0, vmax=255)

        # Title uses the dynamically pulled names
        ax.set_title(f"CH {i}: {channel_names[i]}")

        # Add a colorbar scaled strictly to the subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")  # Hide tick marks for cleaner look

    # 4. Hide any leftover empty subplots
    for j in range(num_channels, len(axes)):
        axes[j].axis("off")

    # 5. Save and aggressively clear memory
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.clf()
    plt.close(fig)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Vectorized PhysiCell runner with CPU pinning."
    )
    parser.add_argument(
        "settingxml", nargs="?", default="config/PhysiCell_settings.xml"
    )
    parser.add_argument("settingcells", nargs="?", default="config/cells.csv")
    parser.add_argument("-m", "--max_time", type=float, default=6400.0)
    parser.add_argument("-n", "--num_envs", type=int, default=9)
    parser.add_argument("-t", "--rl_threads", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=3)
    parser.add_argument("-tt", "--total_timesteps", type=int, default=1e5)
    args = parser.parse_args()
    params = {
        "tumor": {"correlation_length": 45, "threshold": 0.55, "number_cells": 64},
        "t_cell": {"correlation_length": 45, "threshold": 0.55, "number_cells": 8},
        "macrophage": {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
    }
    # ---- Unified nested configuration ----
    cfg = {
        "simulation": {
            "max_time": args.max_time,
            "seed": args.seed,
        },
        "vectorization": {
            "num_envs": args.num_envs,
            "rl_threads": args.rl_threads,
        },
        "model": {
            "id": "physigym/ModelPhysiCellEnv-v0",
            "settingxml": args.settingxml,
            "settingcells": args.settingcells,
            # NOTE: "targeted" builds the 4-var action space (dose,x,y,radius)
            # the wrapper's list_variable_name below expects. "full" would give
            # a dose-only space and raise KeyError: 'drug_1_x' in the wrapper.
            "action_mode": "targeted",
            "output_dir": "./new_wrapper_output_data",
            "figsize": (6, 6),
            "observation_mode": "scalars_macrophages",  # "img_mc_cells_substrates",
            "render_mode": None,
            "verbose": False,
            "img_rgb_grid_size_x": 64,
            "img_rgb_grid_size_y": 64,
            "img_mc_grid_size_x": 64,
            "img_mc_grid_size_y": 64,
            "normalization_factor": 512,
        },
        "wrapper": {
            "list_variable_name": [
                "drug_1_dose",
                "drug_1_x",
                "drug_1_y",
                "drug_1_radius",
            ],
            "w_cell": 0.3,
        },
        "generation": {
            "x_min": 0,
            "x_max": 64,
            "y_min": 0,
            "y_max": 64,
            "params": params,  # number of tumor cells for the initial state
            "seed": args.seed,  # seed
            "mode_train": "network_field",
            "mode_test": ["rectangle", "circular", "network_field"],
        },
        "rl": {"total_timesteps": 25000},
    }

    # ── Publication hyperparameters to validate ─────────────────
    ACTION_REPEAT = 6  # same action held for N steps
    DELTA_X = 0.15  # max x displacement per decision step
    DELTA_Y = 0.15  # max y displacement per decision step
    DELTA_RADIUS = 0.05  # max radius change per decision step
    DELTA_DOSE = 1.0  # unconstrained
    W_SMOOTH = 0.02  # smoothness penalty weight

    cfg["wrapper"]["w_smooth"] = W_SMOOTH
    cfg["wrapper"]["action_delta_max"] = [DELTA_DOSE, DELTA_X, DELTA_Y, DELTA_RADIUS]

    env = test_make_physigym_env(cfg)
    env.reset()
    env.generate_physicell_data = True

    print(
        f"\n[HYPERPARAM CHECK]\n"
        f"  action_repeat  = {ACTION_REPEAT}\n"
        f"  delta [dose,x,y,r] = [{DELTA_DOSE}, {DELTA_X}, {DELTA_Y}, {DELTA_RADIUS}]\n"
        f"  w_smooth       = {W_SMOOTH}\n"
    )

    # ── Per-episode accumulators ─────────────────────────────────
    ep = 0
    ep_reward = 0.0
    ep_dose = 0.0
    ep_steps = 0
    prev_action = None
    deltas_x = []
    deltas_y = []
    deltas_r = []

    MAX_STEPS = 50_000
    step = 0
    while step < MAX_STEPS:
        # random action in [0,1]^4
        raw_action = np.random.rand(4).astype(np.float32)

        # ── action repeat: accumulate reward over N sub-steps ───
        acc_reward = 0.0
        last_obs, last_info, last_term, last_trunc = None, {}, False, False
        for _rep in range(ACTION_REPEAT):
            obs, reward, terminated, truncated, info = env.step(raw_action)
            acc_reward += reward
            last_obs, last_info, last_term, last_trunc = (
                obs,
                info,
                terminated,
                truncated,
            )
            step += 1
            if terminated or truncated:
                break

        # ── track what the wrapper actually executed (after delta-clip) ─
        # The wrapper stores the clipped action in _action_history
        if env._action_history:
            executed = env._action_history[-1]  # [dose, x, y, radius]
            if prev_action is not None:
                deltas_x.append(abs(float(executed[1]) - float(prev_action[1])))
                deltas_y.append(abs(float(executed[2]) - float(prev_action[2])))
                deltas_r.append(abs(float(executed[3]) - float(prev_action[3])))
            prev_action = executed.copy()

        ep_reward += acc_reward
        ep_dose += float(last_info.get("dose_spent", 0.0))
        ep_steps += 1

        if last_term or last_trunc:
            ep += 1
            max_dx = max(deltas_x) if deltas_x else 0.0
            max_dy = max(deltas_y) if deltas_y else 0.0
            max_dr = max(deltas_r) if deltas_r else 0.0
            mean_dx = np.mean(deltas_x) if deltas_x else 0.0
            mean_dy = np.mean(deltas_y) if deltas_y else 0.0

            # ── constraint check ────────────────────────────────
            ok_x = max_dx <= DELTA_X + 1e-4
            ok_y = max_dy <= DELTA_Y + 1e-4
            ok_r = max_dr <= DELTA_RADIUS + 1e-4

            print(
                f"ep={ep:3d}  steps={ep_steps:4d}  reward={ep_reward:+7.3f}  "
                f"dose={ep_dose:.3f}  tumor={last_info.get('number_tumor', '?')}\n"
                f"        delta x  max={max_dx:.4f}  mean={mean_dx:.4f}  "
                f"PASS={'✓' if ok_x else '✗ VIOLATED'} (limit={DELTA_X})\n"
                f"        delta y  max={max_dy:.4f}  mean={mean_dy:.4f}  "
                f"PASS={'✓' if ok_y else '✗ VIOLATED'} (limit={DELTA_Y})\n"
                f"        delta r  max={max_dr:.4f}  "
                f"PASS={'✓' if ok_r else '✗ VIOLATED'} (limit={DELTA_RADIUS})\n"
                f"        action_autocorr_lag1={last_info.get('action_autocorr_lag1', 'n/a')}"
            )

            # reset
            ep_reward = 0.0
            ep_dose = 0.0
            ep_steps = 0
            prev_action = None
            deltas_x.clear()
            deltas_y.clear()
            deltas_r.clear()
            env.reset()
            env.generate_physicell_data = True

    # run_vectorized(cfg)
