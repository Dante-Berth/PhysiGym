"""
run_physigym_tip_sac.py
Synchronous Soft Actor-Critic (SAC) for PhysiGym.

Removed: multiprocessing, queues, async actor process, stop_event.
Added:   single env loop, clean separation of collection / update phases.
"""

import argparse
import random
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
import wandb
from tqdm import tqdm

from vectorized_tip import vec_envs
from nn_tip import Actor, QNetwork
from rb_tip import ReplayBuffer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def obs_to_pyg(obs_dict: dict, device: torch.device) -> Batch:
    """Convert a batched dict-of-arrays observation into a PyG Batch."""
    graphs = []
    B = obs_dict["node_features"].shape[0]
    for i in range(B):
        node_mask = obs_dict["node_mask"][i] > 0.5
        edge_mask = obs_dict["edge_mask"][i] > 0.5

        x = obs_dict["node_features"][i][node_mask]
        edge_index = obs_dict["edge_index"][i][:, edge_mask]
        edge_attr = obs_dict["edge_attr"][i][edge_mask]

        g = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        graphs.append(g)

    return Batch.from_data_list(graphs).to(device)


def build_env_info(envs, d_arg: dict) -> dict:
    """Extract env metadata needed by networks and replay buffer."""
    return {
        "action_space_shape": envs.action_space.shape,
        "observation_space_shape": envs.observation_space.shape,
        "observation_mode": d_arg["model"]["observation_mode"],
        "x_min": envs.get_attr("x_min")[0],
        "x_max": envs.get_attr("x_max")[0],
        "y_min": envs.get_attr("y_min")[0],
        "y_max": envs.get_attr("y_max")[0],
        "action_space_high": envs.action_space.high,
        "action_space_low": envs.action_space.low,
        "observation_space_dtype": envs.observation_space.dtype,
        "is_graph": "graph" in d_arg["model"]["observation_mode"],
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_sac(d_arg: dict) -> None:
    # --- Device & seeds ---
    device = torch.device(
        "cuda" if d_arg["simulation"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    seed = d_arg["simulation"]["seed"] or 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Environments ---
    envs = vec_envs(d_arg)
    obs = envs.reset()
    d_arg_env = build_env_info(envs, d_arg)
    d_arg["env"] = d_arg_env
    is_graph = d_arg_env["is_graph"]
    num_envs = envs.num_envs

    # --- Replay buffer ---
    rb = ReplayBuffer(
        state_dim=d_arg_env["observation_space_shape"],
        action_dim=d_arg_env["action_space_shape"],
        device=device,
        buffer_size=d_arg["rl"]["buffer_size"],
        batch_size=d_arg["rl"]["batch_size"],
        state_type=d_arg_env["observation_space_dtype"],
        is_graph=is_graph,
    )

    # --- Networks ---
    actor = Actor(d_arg_env, d_arg["neural_architecture_image"]).to(device)
    qf1   = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(device)
    qf2   = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(device)
    qf1_target = deepcopy(qf1)
    qf2_target = deepcopy(qf2)

    q_optimizer     = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["rl"]["q_lr"]
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=d_arg["rl"]["policy_lr"])

    # --- Entropy / alpha ---
    if d_arg["rl"]["autotune"]:
        target_entropy = -float(np.prod(d_arg_env["action_space_shape"]))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=d_arg["rl"]["q_lr"])
        alpha = log_alpha.exp().item()
    else:
        alpha = float(d_arg["rl"]["alpha"])
        log_alpha = None
        alpha_optim = None
        target_entropy = None

    # --- Logging ---
    output_dir = d_arg["model"]["output_dir"]
    writer = SummaryWriter(log_dir=output_dir)
    if d_arg["simulation"]["wandb_track"]:
        run = wandb.init(
            project=d_arg["wandb"].get("project", "SAC_TIP"),
            name=Path(output_dir).name,
            config=d_arg,
        )
        run.define_metric("charts/*", step_metric="global_step")

    # --- Training state ---
    tau               = d_arg["rl"]["tau"]
    gamma             = d_arg["rl"]["gamma"]
    total_timesteps   = d_arg["rl"]["total_timesteps"]
    learning_starts   = d_arg["rl"]["learning_starts"]
    policy_frequency  = d_arg["rl"]["policy_frequency"]
    target_freq       = d_arg["rl"]["target_network_frequency"]
    num_loops         = d_arg["rl"]["num_loops"]
    begin_time        = time.time()

    episode_returns = np.zeros(num_envs, dtype=np.float64)
    global_step     = 0
    grad_steps      = 0

    pbar = tqdm(total=total_timesteps)

    try:
        while global_step < total_timesteps:
            pbar.update(global_step - pbar.n)

            # ----------------------------------------------------------------
            # 1. Collect one step from all envs
            # ----------------------------------------------------------------
            if global_step < learning_starts:
                actions = np.array(
                    [envs.action_space.sample() for _ in range(num_envs)],
                    dtype=np.float32,
                )
            else:
                with torch.no_grad():
                    if is_graph:
                        obs_nn = obs_to_pyg(obs, device)
                    else:
                        obs_nn = torch.tensor(obs, dtype=torch.float32, device=device)
                    actions_t, _, _ = actor.get_action(obs_nn)
                    actions = actions_t.cpu().numpy()

            next_obs, rewards, dones, infos = envs.step(actions)

            # Handle full env death
            if all(info.get("disabled", False) for info in infos):
                print("[SAC] All envs dead — restarting VecEnv")
                try:
                    envs.close()
                except Exception:
                    pass
                envs = vec_envs(d_arg)
                obs  = envs.reset()
                num_envs        = envs.num_envs
                episode_returns = np.zeros(num_envs, dtype=np.float64)
                continue

            episode_returns += rewards.astype(np.float64)
            global_step     += num_envs - len(envs.dead_envs)

            # Store transitions & log finished episodes
            for i in range(num_envs):
                if i in envs.dead_envs:
                    continue
                info = infos[i]
                done = dones[i]

                if is_graph:
                    o  = {k: v[i] for k, v in obs.items()}
                    no = {k: v[i] for k, v in next_obs.items()}
                else:
                    o  = obs[i].copy()
                    no = next_obs[i].copy()

                rb.add_batch([(o, actions[i], float(rewards[i]), no, bool(done))])

                if done:
                    if "train_test" not in info or "step_episode" not in info:
                        episode_returns[i] = 0.0
                        continue

                    log_dict = {
                        "global_step": global_step,
                        f"charts/{info['train_test']}_return": float(episode_returns[i]),
                        f"charts/{info['train_test']}_length": int(info["step_episode"]),
                        "charts/grad_steps": grad_steps,
                    }
                    if d_arg["simulation"]["wandb_track"]:
                        run.log(log_dict)
                    else:
                        for tag, val in log_dict.items():
                            if tag != "global_step":
                                writer.add_scalar(tag, val, global_step)

                    episode_returns[i] = 0.0

            obs = next_obs

            # ----------------------------------------------------------------
            # 2. Learning updates (only after warm-up)
            # ----------------------------------------------------------------
            if global_step < max(learning_starts, d_arg["rl"]["batch_size"]):
                continue

            for _ in range(num_loops):
                batch      = rb.sample()
                state      = batch["state"]
                action     = batch["action"]
                reward     = batch["reward"]
                next_state = batch["next_state"]
                done       = batch["done"]

                # --- Critic update ---
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(next_state)
                    q1_next = qf1_target(next_state, next_actions)
                    q2_next = qf2_target(next_state, next_actions)
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    target_q = (
                        reward.flatten()
                        + (1 - done.flatten()) * gamma * min_q_next.squeeze()
                    )

                q1 = qf1(state, action).view(-1)
                q2 = qf2(state, action).view(-1)
                qf_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()
                grad_steps += 1

                # --- Actor & alpha update ---
                if grad_steps % policy_frequency == 0:
                    for _ in range(policy_frequency):
                        pi, log_pi, _ = actor.get_action(state)
                        min_q_pi = torch.min(qf1(state, pi), qf2(state, pi))
                        actor_loss = (alpha * log_pi - min_q_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if d_arg["rl"]["autotune"]:
                            alpha_loss = (
                                -log_alpha.exp() * (log_pi + target_entropy).detach()
                            ).mean()
                            alpha_optim.zero_grad()
                            alpha_loss.backward()
                            alpha_optim.step()
                            alpha = log_alpha.exp().item()

                # --- Target network soft update ---
                if grad_steps % target_freq == 0:
                    for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
                    for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    except KeyboardInterrupt:
        print("Interrupted — shutting down.")

    finally:
        pbar.close()
        try:
            envs.close()
        except Exception:
            pass
        writer.close()
        if d_arg["simulation"]["wandb_track"]:
            wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting synchronous SAC for PhysiGym...")

    parser = argparse.ArgumentParser(
        prog="run_physigym_tip_sac",
        description="Synchronous SAC with PhysiCell + PyG graph support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # File / Environment
    parser.add_argument("--settingxml",   default="config/PhysiCell_settings.xml")
    parser.add_argument("--settingcells", default="config/cells.csv")
    parser.add_argument("--seed",         type=int,   default=1)
    parser.add_argument("--gpu",          default="true")

    # Observation & network
    parser.add_argument("--observation_mode",          default="transformer_nodes")
    parser.add_argument("--neural_architecture_image", default="impala")

    # Training
    parser.add_argument("--max_time_episode",      type=float, default=10800.0)
    parser.add_argument("--learning_starts",       type=int,   default=5_000)
    parser.add_argument("--total_timesteps",       type=int,   default=400_000)
    parser.add_argument("--rl_threads",            type=int,   default=4)
    parser.add_argument("--num_envs",              type=int,   default=3)
    parser.add_argument("--buffer_size",           type=int,   default=300_000)
    parser.add_argument("--batch_size_multiplier", type=int,   default=64)

    # Experiment metadata
    parser.add_argument("--name",   default="TME_V2")
    parser.add_argument("--wandb",  default="true")
    parser.add_argument("--entity", default="corporate-manu-sureli")

    # Cell initialisation
    parser.add_argument("--tumor",               type=int, default=128)
    parser.add_argument("--Macrophage",          type=int, default=32)
    parser.add_argument("--T_cells",             type=int, default=32)
    parser.add_argument("--frequence_episode_test", type=float, default=None)
    parser.add_argument("--img_mc_grid_size",    type=int, default=64)

    args = parser.parse_args()

    i_seed  = None if str(args.seed).lower() == "none" else int(args.seed)
    b_gpu   = args.gpu.lower().startswith("t")
    b_wandb = args.wandb.lower().startswith("t")

    params = {
        "tumor":     {"correlation_length": 45, "threshold": 0.55, "number_cells": args.tumor},
        "Macrophage":{"correlation_length": 45, "threshold": 0.55, "number_cells": args.Macrophage},
        "T_cell":    {"correlation_length": 45, "threshold": 0.55, "number_cells": args.T_cells},
    }

    d_arg = {
        "simulation": {
            "name": args.name,
            "cuda": b_gpu,
            "wandb_track": b_wandb,
            "seed": i_seed,
            "max_time": args.max_time_episode,
        },
        "vectorization": {
            "num_envs": args.num_envs,
            "rl_threads": args.rl_threads,
        },
        "wandb": {
            "entity": args.entity,
            "project": "SAC_TME_Tcells",
            "sync_tensorboard": True,
            "monitor_gym": True,
            "save_code": True,
        },
        "rl": {
            "total_timesteps": args.total_timesteps,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size_multiplier * args.num_envs,
            "learning_starts": args.learning_starts,
            "policy_frequency": 2,
            "target_network_frequency": 1,
            "autotune": True,
            "alpha": 0.05,
            "tau": 0.005,
            "q_lr": 3e-4,
            "policy_lr": 3e-4,
            "gamma": 0.99,
            "num_loops": 3,
        },
        "wrapper": {
            "list_variable_name": ["drug_1"],
            "w_cell": 0.7,
            "w_increase": 0.2,
            "w_amount": 0.1,
            "frequence_episode_test": 4,
        },
        "model": {
            "id": "physigym/ModelPhysiCellEnv-v0",
            "settingxml": args.settingxml,
            "settingcells": args.settingcells,
            "cell_type_cmap": {
                "tumor": "yellow",
                "cell_1": "green",
                "cell_2": "navy",
                "other_tissue": "red",
            },
            "figsize": (6, 6),
            "observation_mode": args.observation_mode,
            "render_mode": None,
            "verbose": False,
            "img_rgb_grid_size_x": args.img_mc_grid_size,
            "img_rgb_grid_size_y": args.img_mc_grid_size,
            "img_mc_grid_size_x": args.img_mc_grid_size,
            "img_mc_grid_size_y": args.img_mc_grid_size,
            "normalization_factor": args.tumor,
        },
        "neural_architecture_image": args.neural_architecture_image,
        "generation": {
            "params": params,
            "seed": i_seed,
            "mode_train": ["network_field", "rectangle"],
            "mode_test": ["random", "circular"],
        },
    }

    d_arg["model"]["output_dir"] = (
        f"data/{d_arg['simulation']['name']}"
        f"_{d_arg['simulation']['seed']}"
        f"_{d_arg['model']['observation_mode']}"
        f"_{int(time.time())}"
    )

    run_sac(d_arg=d_arg)