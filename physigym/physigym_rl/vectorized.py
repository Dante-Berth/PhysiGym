#!/usr/bin/env python3
"""
title: run_physigym_vectorized_combined.py

description:
    Run multiple PhysiCell simulations in parallel with Gymnasium + SubprocVecEnv.
    Each environment:
      - uses a unique XML config and output folder
      - runs on its own CPU core range (affinity pinned)
      - is wrapped in PhysiCellModelWrapper for simplified Box actions

author: Alexandre Bertin (with ChatGPT)
date: 2025
"""

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

from wrapper_tip import (
    PhysiCellModelWrapper,
) 
import sys
import physigym
from extending import physicell


# ============================================================
# Global Thread Splitting
# ============================================================
def configure_thread_splitting(rl_threads: int):
    """
    Configure threads globally for the RL side (PyTorch) and
    leave remaining threads for the PhysiCell simulations.
    """
    import torch

    total = psutil.cpu_count(logical=True)
    rl_threads = max(1, rl_threads)
    rl_threads = min(rl_threads, total - 1)

    # RL / PyTorch threading
    torch.set_num_threads(rl_threads)
    os.environ["OMP_NUM_THREADS"] = str(rl_threads)
    os.environ["MKL_NUM_THREADS"] = str(rl_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(rl_threads)

    print(f"[ThreadSplit] RL threads = {rl_threads} / {total} total cores")
    print(f"[ThreadSplit] Remaining = {total - rl_threads} for PhysiCell envs")

    return rl_threads


# ============================================================
# Helper: CPU affinity per environment
# ============================================================
def assign_cpu_affinity(env_id: int, threads_per_env: int, offset_threads: int):
    os.environ["OMP_NUM_THREADS"] = str(threads_per_env)
    total_cores = psutil.cpu_count(logical=True)
    start = env_id * threads_per_env + offset_threads
    end = min(start + threads_per_env, total_cores)
    core_list = list(range(start, end))
    try:
        psutil.Process().cpu_affinity(core_list)
        print(f"[Affinity] Env {env_id}: pinned to cores {core_list}")
    except Exception as e:
        print(f"[Affinity] Warning: failed to pin Env {env_id}: {e}")


# ============================================================
# Helper: Environment factory
# ============================================================
def make_physigym_env(env_id: int, cfg: dict):
    """
    cfg structure:
    {
        "simulation": {...},
        "vectorization": {...},
        "model": {...},
        "wrapper": {...}
    }
    """

    sim_cfg = cfg["simulation"]
    vect_cfg = cfg["vectorization"]
    model_cfg = cfg["model"]
    wrapper_cfg = cfg["wrapper"]
    generation_cfg = cfg["generation"]

    base_xml = model_cfg["settingxml"]
    base_cells = model_cfg["settingcells"]
    model_cfg_copy = model_cfg.copy()
    threads_per_env = vect_cfg["threads_per_env"]
    seed = sim_cfg["seed"]
    master_seed = seed if seed is not None else 42
    rng = np.random.default_rng(master_seed)
    env_xml = f"config/PhysiCell_settings_env{env_id}.xml"
    env_cells = f"config/cells_{env_id}.csv"
    if not os.path.exists(env_xml):
        shutil.copy(base_xml, env_xml)
    if not os.path.exists(env_cells):
        shutil.copy(base_cells, env_cells)
    if model_cfg_copy["output_dir"] is None:
        model_cfg_copy["output_dir"] = "output"
    del model_cfg_copy["settingcells"]
    rl_threads = vect_cfg["rl_threads"]

    def _init():
        assign_cpu_affinity(env_id, threads_per_env, offset_threads=rl_threads)

        # Modify XML for this env
        tree = etree.parse(env_xml)
        root = tree.getroot()
        root.xpath("//overall/max_time")[0].text = str(sim_cfg["max_time"])
        root.xpath("//parallel/omp_num_threads")[0].text = str(1) #str(threads_per_env)
        root.xpath("//save/folder")[0].text = os.path.join(
            model_cfg_copy["output_dir"], f"env{env_id}"
        )
        root.xpath("//save/full_data/enable")[0].text = "false"
        root.xpath("//save/SVG/enable")[0].text = "false"
        root.xpath("//initial_conditions/cell_positions/filename")[
            0
        ].text = f"cells_{env_id}.csv"
        tree.write(env_xml, pretty_print=True)
        model_cfg_copy["settingxml"] = env_xml

        del model_cfg_copy["output_dir"]
        # if env_id != 0:
        #    wrapper_cfg["frequency_save_data"] = None
        # Create the base PhysiCell environment
        env = gym.make(**model_cfg_copy)
        # Wrap it for simplified action and custom reward
        env = PhysiCellModelWrapper(env, **wrapper_cfg)
        generation_cfg["seed"] = int(rng.integers(0, 2**12 - 1)) + env_id
        env.reset(generation_cfg=generation_cfg)
        return env

    return _init


def vec_envs(cfg: dict):
    vect_cfg = cfg["vectorization"]
    num_envs = vect_cfg["num_envs"]
    rl_threads = vect_cfg[
        "rl_threads"
    ]  # configure_thread_splitting(vect_cfg["rl_threads"])
    threads_per_env = (psutil.cpu_count(logical=True) - rl_threads) // num_envs
    cfg["vectorization"]["threads_per_env"] = threads_per_env
    print(f"[INFO] Launching {num_envs} envs × {threads_per_env} threads each")
    env_fns = [make_physigym_env(i, cfg) for i in range(num_envs)]
    return ResilientSubprocVecEnv(
        env_fns=env_fns, start_method="spawn"
    )  # SubprocVecEnv(env_fns=env_fns, start_method="spawn")


def test_make_physigym_env(cfg: dict, env_id=0):
    assign_cpu_affinity(0, 4, offset_threads=2)
    vect_cfg = cfg["vectorization"]
    num_envs = vect_cfg["num_envs"]
    rl_threads = vect_cfg[
        "rl_threads"
    ]  # configure_thread_splitting(vect_cfg["rl_threads"])
    threads_per_env = (psutil.cpu_count(logical=True) - rl_threads) // num_envs
    cfg["vectorization"]["threads_per_env"] = threads_per_env
    print(f"[INFO] Launching {num_envs} envs × {threads_per_env} threads each")
    sim_cfg = cfg["simulation"]
    vect_cfg = cfg["vectorization"]
    model_cfg = cfg["model"]
    wrapper_cfg = cfg["wrapper"]
    generation_cfg = cfg["generation"]

    base_xml = model_cfg["settingxml"]
    base_cells = model_cfg["settingcells"]
    model_cfg_copy = model_cfg.copy()
    threads_per_env = vect_cfg["threads_per_env"]
    seed = sim_cfg["seed"]
    master_seed = seed if seed is not None else 42
    rng = np.random.default_rng(master_seed)
    env_xml = f"config/PhysiCell_settings_env{env_id}.xml"
    env_cells = f"config/cells_{env_id}.csv"
    if not os.path.exists(env_xml):
        shutil.copy(base_xml, env_xml)
    if not os.path.exists(env_cells):
        shutil.copy(base_cells, env_cells)
    if model_cfg_copy["output_dir"] is None:
        model_cfg_copy["output_dir"] = "output"
    del model_cfg_copy["settingcells"]
    rl_threads = vect_cfg["rl_threads"]
    # Modify XML for this env
    tree = etree.parse(env_xml)
    root = tree.getroot()
    root.xpath("//overall/max_time")[0].text = str(sim_cfg["max_time"])
    root.xpath("//parallel/omp_num_threads")[0].text = str(threads_per_env)
    root.xpath("//save/folder")[0].text = os.path.join(
        model_cfg_copy["output_dir"], f"env{env_id}"
    )
    root.xpath("//save/full_data/enable")[0].text = "false"
    root.xpath("//save/SVG/enable")[0].text = "false"
    root.xpath("//initial_conditions/cell_positions/filename")[
        0
    ].text = f"cells_{env_id}.csv"
    tree.write(env_xml, pretty_print=True)
    model_cfg_copy["settingxml"] = env_xml

    del model_cfg_copy["output_dir"]
    # if env_id != 0:
    #    wrapper_cfg["frequency_save_data"] = None
    # Create the base PhysiCell environment
    env = gym.make(**model_cfg_copy)
    # Wrap it for simplified action and custom reward
    env = PhysiCellModelWrapper(env, **wrapper_cfg)
    generation_cfg["seed"] = int(rng.integers(0, 2**12 - 1)) + env_id
    env.reset(generation_cfg=generation_cfg)
    return env


# ============================================================
# Runner
# ============================================================
def run_vectorized(cfg: dict):
    vect_cfg = cfg["vectorization"]
    num_envs = vect_cfg["num_envs"]
    envs = vec_envs(cfg)
    _ = envs.reset()
    num_envs = envs.num_envs
    time_1 = time.time()
    total = cfg["rl"]["total_timesteps"]
    pbar = tqdm(total=total)
    local_step = 0
    while local_step < total:
        pbar.update(local_step - pbar.n)
        actions = np.array(
            [[0.0] for _ in range(num_envs)],
            dtype=np.float32,
        )
        _, _, _, infos = envs.step(actions)
        local_step += num_envs
        """
        local_step += num_envs - len(envs.dead_envs)
        if all(info.get("disabled", False) for info in infos):
            print("[Actor] All envs dead — restarting VecEnv")

            try:
                envs.close()

            except Exception:
                pass
            del envs
            envs = vec_envs(cfg)
            _ = envs.reset()

            num_envs = envs.num_envs
        """
    envs.close()
    return round(time.time() - time_1, 2)


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
    parser.add_argument("-m", "--max_time", type=float, default=12900.0)
    parser.add_argument("-n", "--num_envs", type=int, default=9)
    parser.add_argument("-t", "--rl_threads", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=3)
    parser.add_argument("-tt", "--total_timesteps", type=int, default=1e5)
    args = parser.parse_args()
    params = {
        "tumor": {"correlation_length": 45, "threshold": 0.55, "number_cells": 128},
        "T_cell": {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
        "Macrophage": {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
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
            "output_dir": "./new_wrapper_output_data",
            "figsize": (6, 6),
            "observation_mode": "img_mc_cells_substrates",  # "img_mc_cells_substrates",
            "render_mode": None,
            "verbose": False,
            "img_rgb_grid_size_x": 64,
            "img_rgb_grid_size_y": 64,
            "img_mc_grid_size_x": 64,
            "img_mc_grid_size_y": 64,
            "normalization_factor": 512,
        },
        "wrapper": {
            "list_variable_name": ["drug_1"],
            "w_cell": 0.7,
            "w_increase": 0.2,
            "w_amount": 0.1,
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
    env = test_make_physigym_env(cfg)
    env.reset()
    run_vectorized(cfg)
