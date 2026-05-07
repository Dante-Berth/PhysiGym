import argparse
import os
import random
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
import wandb

from tqdm import tqdm

# Your project imports
from vectorized_tip import vec_envs
from nn_tip import Actor, QNetwork
from rb_tip import ReplayBuffer

import queue
from torch.multiprocessing import Event, Queue


# --------------------------------------------------------------
# Helper: convert dict-of-arrays → PyG Batch
# --------------------------------------------------------------
def obs_to_pyg(obs_dict, device):
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
        g.batch = torch.full((x.shape[0],), i, dtype=torch.long)
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    return batch.to(device)


# --------------------------------------------------------------
# Actor process — uses shared_actor for zero-copy policy sync
# --------------------------------------------------------------
def actor_process(
    shared_actor,  # shared-memory actor: always fresh, zero copy, zero lag
    sample_queue,
    stats_queue,
    d_arg,
    stop_event,
    env_info_queue,
):
    print(d_arg)
    envs = vec_envs(d_arg)

    begin_time = time.time()
    obs = envs.reset()

    is_graph = "graph" in d_arg["model"]["observation_mode"]

    d_arg_env = {
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
        "is_graph": is_graph,
    }

    env_info_queue.put(d_arg_env)

    # Use shared_actor directly — no local copy, no queue, always up-to-date
    shared_actor.eval()

    num_envs = envs.num_envs
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    local_step = 0

    while not stop_event.is_set():
        if local_step <= d_arg["rl"]["learning_starts"]:
            actions = np.array(
                [envs.action_space.sample() for _ in range(num_envs)],
                dtype=np.float32,
            )
        else:
            with torch.no_grad():
                if is_graph:
                    pyg_batch = obs_to_pyg(obs, "cpu")
                    actions_tensor, _, _ = shared_actor.get_action(pyg_batch)
                else:
                    x = torch.from_numpy(obs).cpu()
                    actions_tensor, _, _ = shared_actor.get_action(x)
                actions = actions_tensor.cpu().numpy()

        next_obs, rewards, dones, infos = envs.step(actions)

        # Handle dead envs
        if all(info.get("disabled", False) for info in infos):
            print("[Actor] All envs dead — restarting VecEnv")
            try:
                envs.close()
            except Exception:
                pass
            del envs
            envs = vec_envs(d_arg)
            obs = envs.reset()
            num_envs = envs.num_envs
            episode_returns = np.zeros(num_envs, dtype=np.float64)
            continue

        episode_returns += rewards.astype(np.float64)
        local_step += num_envs - len(envs.dead_envs)

        batch_samples = []
        for i in range(num_envs):
            if i in envs.dead_envs:
                continue
            info = infos[i]
            done = dones[i]

            if is_graph:
                o = {k: v[i] for k, v in obs.items()}
                no = {k: v[i] for k, v in next_obs.items()}
            else:
                o = obs[i].copy() if isinstance(obs[i], np.ndarray) else obs[i]
                no = (
                    next_obs[i].copy()
                    if isinstance(next_obs[i], np.ndarray)
                    else next_obs[i]
                )

            if done:
                if "train_test" not in info or "step_episode" not in info:
                    continue
                try:
                    stats_queue.put_nowait(
                        {
                            "episode_return": float(episode_returns[i]),
                            "episode_length": int(info["step_episode"]),
                            "step": int(local_step),
                            "timestamp": time.time() - begin_time,
                            "train_test": info["train_test"],
                            "type_mode": info["type_mode"],
                        }
                    )
                except queue.Full:
                    pass
                episode_returns[i] = 0.0

            if info["train_test"] == "train":
                batch_samples.append(
                    (o, actions[i], float(rewards[i]), no, bool(dones[i]))
                )

        if batch_samples:
            try:
                sample_queue.put_nowait(batch_samples)
            except queue.Full:
                pass  # drop batch if learner is overloaded

        obs = next_obs

    try:
        envs.close()
    except Exception:
        pass


# --------------------------------------------------------------
# Main learner
# --------------------------------------------------------------
def run_async_sac(d_arg):
    # ── Sliding-window return trackers ──────────────────────────
    return_buffers = {
        "train": deque(maxlen=50),
        "test": deque(maxlen=50),
    }

    # ── Device & seeds ──────────────────────────────────────────
    device = torch.device(
        "cuda" if d_arg["simulation"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    seed = d_arg["simulation"]["seed"] or 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Inter-process queues ────────────────────────────────────
    # actor_queue REMOVED — policy sync now via shared memory
    sample_queue = mp.Queue(maxsize=10_000)
    stats_queue = mp.Queue(maxsize=1_000)
    env_info_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    # ── Build actor in shared memory BEFORE spawning process ───
    # We need d_arg_env first → use a temporary actor proc just to get env info,
    # then rebuild. Simpler: spawn the proc, get env info, build shared actor,
    # send it back. Here we use a two-phase approach with a ready_event.
    #
    # Phase 1: spawn a probe process to get env info only
    probe_queue = mp.Queue(maxsize=1)

    def _probe(d_arg, q):
        envs = vec_envs(d_arg)
        envs.reset()
        info = {
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
        q.put(info)
        envs.close()

    probe_proc = mp.Process(target=_probe, args=(d_arg, probe_queue), daemon=True)
    probe_proc.start()
    d_arg_env = probe_queue.get()
    probe_proc.join()
    d_arg["env"] = d_arg_env

    # ── Build networks ──────────────────────────────────────────
    actor = Actor(d_arg_env, d_arg["neural_architecture_image"]).to(device)
    qf1 = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(device)
    qf2 = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(device)

    # Warm-up forward pass
    if d_arg_env["is_graph"]:
        dummy_graph = Data(
            x=torch.zeros((1, d_arg_env["node_feature_dim"]), dtype=torch.float32),
            edge_index=torch.zeros((2, 1), dtype=torch.long),
            edge_attr=torch.zeros((1, 1), dtype=torch.float32),
        )
        dummy_state = Batch.from_data_list([dummy_graph]).to(device)
    else:
        dummy_state = torch.zeros(
            (1, *d_arg_env["observation_space_shape"]),
            device=device,
            dtype=torch.float32,
        )

    with torch.no_grad():
        actions_tensor, _, _ = actor.get_action(dummy_state)
        _ = qf1(dummy_state, actions_tensor)
        _ = qf2(dummy_state, actions_tensor)

    # Optional: compile for speed (PyTorch 2.0+)
    if d_arg.get("compile", False):
        actor = torch.compile(actor)
        qf1 = torch.compile(qf1)
        qf2 = torch.compile(qf2)

    qf1_target = deepcopy(qf1).to(device)
    qf2_target = deepcopy(qf2).to(device)

    # ── Shared-memory actor for zero-copy policy sync ───────────
    # Actor process reads weights directly from shared memory.
    # No queue, no serialization, no lag.
    shared_actor = Actor(d_arg_env, d_arg["neural_architecture_image"]).cpu()
    shared_actor.load_state_dict(actor.state_dict())
    shared_actor.share_memory()  # ← key call: weights live in shared RAM
    shared_actor.eval()

    # ── Spawn actor process ─────────────────────────────────────
    actor_proc = mp.Process(
        target=actor_process,
        args=(
            shared_actor,
            sample_queue,
            stats_queue,
            d_arg,
            stop_event,
            env_info_queue,
        ),
        daemon=False,
    )
    actor_proc.start()
    # env_info_queue now redundant (we probed above), but actor still sends — drain it
    try:
        env_info_queue.get(timeout=60)
    except Exception:
        pass

    # ── Replay buffer ───────────────────────────────────────────
    rb = ReplayBuffer(
        state_dim=d_arg_env["observation_space_shape"],
        action_dim=d_arg_env["action_space_shape"],
        device=device,
        buffer_size=d_arg["rl"]["buffer_size"],
        batch_size=d_arg["rl"]["batch_size"],
        state_type=d_arg_env["observation_space_dtype"],
        is_graph=d_arg_env["is_graph"],
    )

    # ── Optimizers ──────────────────────────────────────────────
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=d_arg["rl"]["q_lr"],
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=d_arg["rl"]["policy_lr"])

    # Alpha (entropy)
    if d_arg["rl"]["autotune"]:
        target_entropy = -float(np.prod(d_arg_env["action_space_shape"]))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=d_arg["rl"]["q_lr"])
        alpha = log_alpha.exp().item()
    else:
        alpha = float(d_arg["rl"]["alpha"])

    # ── Logging ─────────────────────────────────────────────────
    output_dir = d_arg["model"]["output_dir"]
    writer = SummaryWriter(log_dir=output_dir)

    if d_arg["simulation"]["wandb_track"]:
        run = wandb.init(
            project=d_arg["wandb"]["project"] if "wandb" in d_arg else "SAC_ASYNC_TIP",
            name=Path(output_dir).name,
            config=d_arg,
        )
        run.define_metric("charts/*", step_metric="samples_drained")

    tau = d_arg["rl"]["tau"]
    total_timesteps = d_arg["rl"]["total_timesteps"]
    utd_ratio = d_arg["rl"].get("utd_ratio", 1)  # grad steps per env step
    learning_starts = d_arg["rl"]["learning_starts"]
    batch_size = d_arg["rl"]["batch_size"]

    # ── Training loop ───────────────────────────────────────────
    try:
        print("Starting training loop...")
        pbar = tqdm(total=total_timesteps)

        drained = 0
        grad_steps = 0

        while drained < total_timesteps:
            pbar.update(drained - pbar.n)

            # ── 1) Drain sample queue ────────────────────────────
            local_batch = []

            # Block briefly on first item to avoid pure busy-spin
            try:
                first = sample_queue.get(timeout=0.005)
                local_batch.extend(first if isinstance(first, list) else [first])
            except queue.Empty:
                pass

            # Drain the rest without blocking
            while True:
                try:
                    item = sample_queue.get_nowait()
                    local_batch.extend(item if isinstance(item, list) else [item])
                except queue.Empty:
                    break

            if local_batch:
                rb.add_batch(local_batch)
                drained += len(local_batch)

                # ── Push updated weights to shared actor ─────────
                # Copy learner weights → shared actor (in-place, zero alloc)
                with torch.no_grad():
                    for src, dst in zip(actor.parameters(), shared_actor.parameters()):
                        dst.data.copy_(src.data.cpu())

            # ── 2) Log episode stats ─────────────────────────────
            while not stats_queue.empty():
                try:
                    stat = stats_queue.get_nowait()
                except queue.Empty:
                    break

                split = stat["train_test"]  # "train" or "test"
                return_buffers[split].append(stat["episode_return"])

                log_dict = {
                    f"charts/{split}_return_raw": stat["episode_return"],
                    f"charts/{split}_return_mean50": np.mean(return_buffers[split]),
                    f"charts/{split}_return_std": np.std(return_buffers[split]),
                    "charts/grad_steps": grad_steps,
                }

                # Fixed: single log call, anchored to env steps
                if d_arg["simulation"]["wandb_track"]:
                    run.log(log_dict, step=drained)
                else:
                    for tag, value in log_dict.items():
                        writer.add_scalar(tag, value, drained)

            # ── 3) Wait for enough samples ───────────────────────
            if drained < max(learning_starts, batch_size):
                continue  # no sleep — let the drain loop spin freely

            # ── 4) Gradient updates (UTD-controlled) ─────────────
            # grad steps proportional to new data collected this iteration
            grad_steps_to_do = max(1, int(len(local_batch) * utd_ratio))

            for _ in range(grad_steps_to_do):
                batch = rb.sample()
                next_state = batch["next_state"]
                state = batch["state"]
                action = batch["action"]
                done = batch["done"]
                reward = batch["reward"]

                # Compute Q targets
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(next_state)
                    q1_next = qf1_target(next_state, next_actions)
                    q2_next = qf2_target(next_state, next_actions)
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    next_q = (
                        reward.flatten()
                        + (1 - done.flatten())
                        * d_arg["rl"]["gamma"]
                        * min_q_next.squeeze()
                    )

                q1 = qf1(state, action).view(-1)
                q2 = qf2(state, action).view(-1)
                qf1_loss = F.mse_loss(q1, next_q)
                qf2_loss = F.mse_loss(q2, next_q)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()
                grad_steps += 1

                # Policy & alpha update
                if grad_steps % d_arg["rl"]["policy_frequency"] == 0:
                    for _ in range(d_arg["rl"]["policy_frequency"]):
                        actions_pi, log_pi, _ = actor.get_action(state)
                        q1_pi = qf1(state, actions_pi)
                        q2_pi = qf2(state, actions_pi)
                        actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

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

                # Soft-update target networks
                if grad_steps % d_arg["rl"]["target_network_frequency"] == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )

                # Log training internals periodically
                if grad_steps % 500 == 0 and d_arg["simulation"]["wandb_track"]:
                    run.log(
                        {
                            "charts/qf1_loss": qf1_loss.item(),
                            "charts/qf2_loss": qf2_loss.item(),
                            "charts/actor_loss": actor_loss.item()
                            if grad_steps % d_arg["rl"]["policy_frequency"] == 0
                            else 0,
                            "charts/alpha": alpha,
                        },
                        step=drained,
                    )

            # NOTE: time.sleep removed — gradient compute is the natural throttle

    except KeyboardInterrupt:
        print("Interrupted by user — shutting down.")

    finally:
        stop_event.set()
        actor_proc.join(timeout=5.0)
        if actor_proc.is_alive():
            actor_proc.terminate()
            actor_proc.join(timeout=1.0)

        writer.close()
        if d_arg["simulation"]["wandb_track"]:
            wandb.finish()


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Starting asynchronous SAC for PhysiGym...")

    parser = argparse.ArgumentParser(
        prog="run_physigym_episodes",
        description="Asynchronous SAC with PhysiCell + PyG graph support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === File / Environment Settings ===
    parser.add_argument(
        "--settingxml",
        default="config/PhysiCell_settings.xml",
        help="Path to PhysiCell settings XML file",
    )
    parser.add_argument(
        "--settingcells",
        default="config/cells.csv",
        help="Path to initial cell CSV",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--gpu", type=str, default="true", help="Use GPU? (true/false)")

    # === Observation & Neural Network ===
    parser.add_argument(
        "--observation_mode",
        default="transformer_nodes",
        help="Observation mode for RL agent",
    )
    parser.add_argument(
        "--neural_architecture_image",
        default="impala",
        help="Neural network architecture for image/graph input",
    )

    # === Training / RL Settings ===
    parser.add_argument(
        "--max_time_episode",
        type=float,
        default=10800.0,
        help="Max simulation time per episode (minutes)",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=5_000,
        help="Environment steps before learning starts",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=int(4e5),
        help="Total environment steps for training",
    )
    parser.add_argument(
        "--rl_threads",
        type=int,
        default=3,
        help="Number of RL threads inside the actor process",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=9,
        help="Parallel PhysiCell instances",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=int(3e5),
        help="Replay buffer size",
    )
    parser.add_argument(
        "--batch_size_multiplier",
        type=int,
        default=64,
        help="Batch size = multiplier × num_envs",
    )
    parser.add_argument(
        "--utd_ratio",
        type=float,
        default=1.0,
        help="Update-to-data ratio: gradient steps per new env step",
    )
    parser.add_argument(
        "--policy_frequency",
        type=int,
        default=2,
        help="Actor update every N critic updates",
    )
    parser.add_argument(
        "--target_network_frequency",
        type=int,
        default=1,
        help="Target network soft-update every N critic updates",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default="false",
        help="torch.compile networks? (true/false, requires PyTorch >= 2.0)",
    )

    # === Experiment Metadata ===
    parser.add_argument("--name", default="TME_V2", help="Experiment name")
    parser.add_argument("--wandb", default="true", help="Log to W&B? (true/false)")
    parser.add_argument(
        "--entity", default="corporate-manu-sureli", help="WandB entity name"
    )

    # === Initialization & Cells ===
    parser.add_argument(
        "--tumor", type=int, default=128, help="Initial tumour cell count"
    )
    parser.add_argument(
        "--Macrophage", type=int, default=32, help="Initial macrophage count"
    )
    parser.add_argument("--T_cells", type=int, default=32, help="Initial T-cell count")
    parser.add_argument(
        "--frequence_episode_test",
        type=int,  # was float, but it's a count of episodes → int makes more sense
        default=4,
        help="Run a test episode every N training episodes",
    )
    parser.add_argument(
        "--img_mc_grid_size",
        type=int,
        default=64,
        help="Grid size for image/MC observations",
    )
    parser.add_argument(
        "--w_cell",
        type=float,
        default=0.3,
        help="Weight of cell-count term in the reward wrapper",
    )

    args = parser.parse_args()

    # ── Type coercions ──────────────────────────────────────────
    i_seed = None if str(args.seed).lower() == "none" else int(args.seed)
    b_gpu = args.gpu.lower().startswith("t")
    b_wandb = args.wandb.lower().startswith("t")
    b_compile = args.compile.lower().startswith("t")

    # ── Sub-dicts ───────────────────────────────────────────────

    d_arg_simulation = {
        "name": args.name,
        "cuda": b_gpu,
        "wandb_track": b_wandb,
        "seed": i_seed,
        "max_time": args.max_time_episode,
    }

    d_arg_wandb = {
        "entity": args.entity,
        "project": "SAC_ASYNC_TME_Tcells",
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
    }

    d_arg_physigym_model = {
        "id": "physigym/ModelPhysiCellEnv-v0",
        "settingxml": args.settingxml,
        "settingcells": args.settingcells,
        "cell_type_cmap": {
            "tumor": "red",
            "t_cell": "blue",
            "macrophage": "green",
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
        # output_dir set AFTER this dict is built (needs name + seed + mode + time)
    }

    d_arg_physigym_wrapper = {
        "list_variable_name": ["drug_1_dose", "drug_1_x", "drug_1_y", "drug_1_radius"],
        "w_cell": args.w_cell,  # FIX: was hardcoded 0.7
        "frequence_episode_test": args.frequence_episode_test,  # FIX: was hardcoded 4, ignoring CLI arg
    }

    d_arg_rl = {
        "total_timesteps": args.total_timesteps,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size_multiplier * args.num_envs,
        "learning_starts": args.learning_starts,
        "utd_ratio": args.utd_ratio,  # FIX: replaces dead num_loops
        "policy_frequency": args.policy_frequency,
        "target_network_frequency": args.target_network_frequency,
        "autotune": True,
        "alpha": 0.05,
        "tau": 0.005,
        "q_lr": 3e-4,
        "policy_lr": 3e-4,
        "gamma": 0.99,
        # num_loops REMOVED — replaced by utd_ratio in the learner
    }

    d_arg_vect = {
        "num_envs": args.num_envs,
        "rl_threads": args.rl_threads,
    }

    params = {
        "tumor": {
            "correlation_length": 45,
            "threshold": 0.55,
            "number_cells": args.tumor,
        },
        "macrophage": {
            "correlation_length": 45,
            "threshold": 0.55,
            "number_cells": args.Macrophage,
        },
        "t_cell": {
            "correlation_length": 45,
            "threshold": 0.55,
            "number_cells": args.T_cells,
        },
    }

    d_arg_generation = {
        "params": params,
        "seed": i_seed,
        "mode_train": ["network_field"],
        "mode_test": ["circular", "rectangle"],
    }

    # ── Assemble final d_arg ────────────────────────────────────
    d_arg = {
        "simulation": d_arg_simulation,
        "vectorization": d_arg_vect,
        "wandb": d_arg_wandb,
        "rl": d_arg_rl,
        "wrapper": d_arg_physigym_wrapper,
        "model": d_arg_physigym_model,
        "neural_architecture_image": args.neural_architecture_image,
        "generation": d_arg_generation,
        "compile": b_compile,
    }

    # output_dir built AFTER d_arg["model"] exists (avoids fragile forward-ref)
    d_arg["model"]["output_dir"] = (
        f"data/"
        f"{d_arg['simulation']['name']}_"
        f"{d_arg['simulation']['seed']}_"
        f"{d_arg['model']['observation_mode']}_"
        f"{int(time.time())}"
    )

    # ── Launch ──────────────────────────────────────────────────
    run_async_sac(d_arg=d_arg)
