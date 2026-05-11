import argparse
import os
import random
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
import threading

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
import itertools

_log_counter = itertools.count(0)


# --------------------------------------------------------------
# Helper: convert dict-of-arrays → PyG Batch (same as your original)
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


def actor_process(
    actor_queue,
    sample_queue,
    stats_queue,
    d_arg,
    stop_event,
    env_info_queue,
):
    # ── identical to your original ──────────────────────────────
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

    actor_local = Actor(
        d_arg_env, d_arg.get("neural_architecture_image", "impala")
    ).cpu()

    if d_arg_env["is_graph"]:
        obs_nn = obs_to_pyg(obs, "cpu")
    else:
        obs_nn = torch.from_numpy(obs).cpu()
        _, _, _ = actor_local.get_action(obs_nn)

    actor_local.eval()

    num_envs = envs.num_envs
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    local_step = 0

    while not stop_event.is_set():
        # Try to fetch a new policy (non-blocking)
        try:
            while True:
                new_params = actor_queue.get_nowait()
                try:
                    actor_local.load_state_dict(new_params)
                except Exception:
                    actor_local.load_state_dict(
                        {k: v.cpu() for k, v in new_params.items()}
                    )
        except queue.Empty:
            pass

        if local_step <= d_arg["rl"]["learning_starts"]:
            actions = np.array(
                [envs.action_space.sample() for _ in range(num_envs)],
                dtype=np.float32,
            )
        else:
            with torch.no_grad():
                if is_graph:
                    pyg_batch = obs_to_pyg(obs, "cpu")
                    actions_tensor, _, _ = actor_local.get_action(pyg_batch)
                else:
                    x = torch.from_numpy(obs).cpu()
                    actions_tensor, _, _ = actor_local.get_action(x)
                actions = actions_tensor.cpu().numpy()

        next_obs, rewards, dones, infos = envs.step(actions)

        restarted = False
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
            restarted = True

        if restarted:
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
                train_test = info.get("train_test", "train")
                step_ep = info.get("step_episode", 0)
                type_mode = info.get("type_mode", "unknown")
                try:
                    stats_queue.put_nowait(
                        {
                            "episode_return": float(episode_returns[i]),
                            "episode_length": int(step_ep),
                            "step": int(local_step),
                            "timestamp": time.time() - begin_time,
                            "train_test": train_test,
                            "type_mode": type_mode,
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
                pass  # drop if learner is overloaded

        obs = next_obs

    try:
        envs.close()
    except Exception:
        pass


def run_async_sac(d_arg):
    # ── Sliding-window return trackers ──────────────────────────
    return_buffers = {
        "train": deque(maxlen=50),
        "test": deque(maxlen=50),
    }

    device = torch.device(
        "cuda" if d_arg["simulation"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    seed = d_arg["simulation"]["seed"] or 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Queues — same as your original ─────────────────────────
    actor_queue = mp.Queue(maxsize=5)
    sample_queue = mp.Queue(maxsize=10_000)
    stats_queue = mp.Queue(maxsize=100)
    env_info_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    actor_proc = mp.Process(
        target=actor_process,
        args=(
            actor_queue,
            sample_queue,
            stats_queue,
            d_arg,
            stop_event,
            env_info_queue,
        ),
        daemon=False,
    )
    actor_proc.start()
    d_arg_env = env_info_queue.get()  # blocks until actor sends
    d_arg["env"] = d_arg_env

    rb = ReplayBuffer(
        state_dim=d_arg_env["observation_space_shape"],
        action_dim=d_arg_env["action_space_shape"],
        device=device,
        buffer_size=d_arg["rl"]["buffer_size"],
        batch_size=d_arg["rl"]["batch_size"],
        state_type=d_arg_env["observation_space_dtype"],
        is_graph=d_arg_env["is_graph"],
    )

    actor = Actor(d_arg_env, d_arg["neural_architecture_image"]).to(device)
    qf1 = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(device)
    qf2 = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(device)

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

    qf1_target = deepcopy(qf1).to(device)
    qf2_target = deepcopy(qf2).to(device)

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=d_arg["rl"]["q_lr"],
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=d_arg["rl"]["policy_lr"])

    if d_arg["rl"]["autotune"]:
        target_entropy = -float(np.prod(d_arg_env["action_space_shape"]))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=d_arg["rl"]["q_lr"])
        alpha = log_alpha.exp().item()
    else:
        alpha = float(d_arg["rl"]["alpha"])

    # send initial policy to actor
    try:
        actor_queue.put_nowait(
            {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        )
    except queue.Full:
        actor_queue.put({k: v.detach().cpu() for k, v in actor.state_dict().items()})

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
    learning_starts = d_arg["rl"]["learning_starts"]
    batch_size = d_arg["rl"]["batch_size"]

    # ------------------------------------------------------------------
    # FIX: drain thread
    # Root cause of the freeze: in the original code draining and gradient
    # steps were sequential. During GPU compute the sample_queue filled up
    # and put_nowait silently dropped all new batches → drained stopped
    # incrementing → tqdm appeared frozen → actor had nowhere to put data.
    #
    # Solution: move draining to a background thread so the queue is
    # always emptied regardless of what the GPU is doing.
    # Only two lines touch the ReplayBuffer (add/sample) so a simple
    # threading.Lock() is sufficient — it is never held during GPU compute.
    # ------------------------------------------------------------------
    drained = 0
    drain_lock = threading.Lock()
    drain_done = threading.Event()

    def drain_worker():
        nonlocal drained
        while not drain_done.is_set():
            local = []
            # small blocking get so we don't busy-spin
            try:
                first = sample_queue.get(timeout=0.01)
                local.extend(first if isinstance(first, list) else [first])
            except queue.Empty:
                continue
            # drain the rest without blocking
            while True:
                try:
                    item = sample_queue.get_nowait()
                    local.extend(item if isinstance(item, list) else [item])
                except queue.Empty:
                    break
            if local:
                with drain_lock:
                    rb.add_batch(local)
                    drained += len(local)

    drain_thread = threading.Thread(target=drain_worker, daemon=True)
    drain_thread.start()

    # ── Training loop ───────────────────────────────────────────
    try:
        print("Starting training loop...")
        pbar = tqdm(total=total_timesteps, dynamic_ncols=True)
        grad_steps = 0

        while drained < total_timesteps:
            pbar.update(drained - pbar.n)
            pbar.set_postfix(
                {
                    "rb": rb.size,
                    "drained": drained,
                    "stats_q": stats_queue.qsize(),
                    "grads": grad_steps,
                },
                refresh=True,
            )

            # ── Log episode stats ────────────────────────────────
            while not stats_queue.empty():
                try:
                    stat = stats_queue.get_nowait()
                except queue.Empty:
                    break

                split = stat["train_test"]
                typemode = stat.get("type_mode", "unknown")
                ep_return = stat["episode_return"]

                return_buffers[split].append(ep_return)

                log_dict = {
                    f"charts/{split}_return_raw": ep_return,
                    f"charts/{split}_{typemode}_return_raw": ep_return,
                    "charts/grad_steps": grad_steps,
                }

                buf = return_buffers[split]
                if len(buf) >= 2:
                    log_dict[f"charts/{split}_return_mean50"] = np.mean(buf)
                    log_dict[f"charts/{split}_return_std"] = np.std(buf)

                if d_arg["simulation"]["wandb_track"]:
                    run.log(log_dict, step=drained)
                else:
                    for tag, value in log_dict.items():
                        writer.add_scalar(tag, value, drained)

            # ── Wait for enough samples ──────────────────────────
            if drained < max(learning_starts, batch_size):
                time.sleep(0.01)
                continue

            # ── Gradient updates (num_loops same as your original) ─
            for _ in range(d_arg["rl"]["num_loops"]):
                with drain_lock:
                    if rb.size < batch_size:
                        break
                    batch = rb.sample()

                next_state = batch["next_state"]
                state = batch["state"]
                action = batch["action"]
                done = batch["done"]
                reward = batch["reward"]

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

                if grad_steps % 500 == 0 and d_arg["simulation"]["wandb_track"]:
                    run.log(
                        {
                            "charts/qf1_loss": qf1_loss.item(),
                            "charts/qf2_loss": qf2_loss.item(),
                            "charts/alpha": alpha,
                        },
                        step=drained,
                    )

            # ── Push policy to actor ─────────────────────────────
            # adaptive frequency: more often early in training
            sync_freq = 16 if grad_steps < 1_000 else 64
            if grad_steps % sync_freq == 0:
                try:
                    actor_queue.put_nowait(
                        {k: v.detach().cpu() for k, v in actor.state_dict().items()}
                    )
                except queue.Full:
                    pass

    except KeyboardInterrupt:
        print("Interrupted by user — shutting down.")

    finally:
        drain_done.set()
        drain_thread.join(timeout=3.0)
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

    parser.add_argument("--settingxml", default="config/PhysiCell_settings.xml")
    parser.add_argument("--settingcells", default="config/cells.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="true")
    parser.add_argument("--observation_mode", default="transformer_nodes")
    parser.add_argument("--neural_architecture_image", default="impala")
    parser.add_argument("--max_time_episode", type=float, default=7200.0)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--total_timesteps", type=int, default=int(5e5))
    parser.add_argument("--rl_threads", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=9)
    parser.add_argument("--buffer_size", type=int, default=int(3e5))
    parser.add_argument("--batch_size_multiplier", type=int, default=64)
    parser.add_argument("--num_loops", type=int, default=3)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--name", default="TME_V2")
    parser.add_argument("--wandb", default="true")
    parser.add_argument("--entity", default="corporate-manu-sureli")
    parser.add_argument("--tumor", type=int, default=128)
    parser.add_argument("--Macrophage", type=int, default=32)
    parser.add_argument("--T_cells", type=int, default=32)
    parser.add_argument("--frequence_episode_test", type=int, default=4)
    parser.add_argument("--img_mc_grid_size", type=int, default=64)
    parser.add_argument("--w_cell", type=float, default=0.3)

    args = parser.parse_args()

    i_seed = None if str(args.seed).lower() == "none" else int(args.seed)
    b_gpu = args.gpu.lower().startswith("t")
    b_wandb = args.wandb.lower().startswith("t")

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
    }

    d_arg_physigym_wrapper = {
        "list_variable_name": ["drug_1_dose", "drug_1_x", "drug_1_y", "drug_1_radius"],
        "w_cell": args.w_cell,
        "frequence_episode_test": args.frequence_episode_test,
    }

    d_arg_rl = {
        "total_timesteps": args.total_timesteps,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size_multiplier * args.num_envs,
        "learning_starts": args.learning_starts,
        "num_loops": args.num_loops,
        "policy_frequency": args.policy_frequency,
        "target_network_frequency": args.target_network_frequency,
        "autotune": True,
        "alpha": 0.05,
        "tau": 0.005,
        "q_lr": 3e-4,
        "policy_lr": 3e-4,
        "gamma": 0.99,
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

    d_arg = {
        "simulation": d_arg_simulation,
        "vectorization": d_arg_vect,
        "wandb": d_arg_wandb,
        "rl": d_arg_rl,
        "wrapper": d_arg_physigym_wrapper,
        "model": d_arg_physigym_model,
        "neural_architecture_image": args.neural_architecture_image,
        "generation": d_arg_generation,
    }

    d_arg["model"]["output_dir"] = (
        f"data/"
        f"{d_arg['simulation']['name']}_"
        f"{d_arg['simulation']['seed']}_"
        f"{d_arg['model']['observation_mode']}_"
        f"{int(time.time())}"
    )

    run_async_sac(d_arg=d_arg)
