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
from custom_modules.physigym.physigym.envs.vectorized import vec_envs
from custom_modules.physigym.physigym.envs.nn import Actor, QNetwork
from custom_modules.physigym.physigym.envs.rb import ReplayBuffer

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

        # action repeat: apply the same action for action_repeat steps,
        # accumulate rewards, use the final next_obs as the transition target
        action_repeat = d_arg["rl"].get("action_repeat", 1)
        accumulated_rewards = np.zeros(num_envs, dtype=np.float32)
        for _rep in range(action_repeat):
            next_obs, rewards, dones, infos = envs.step(actions)
            accumulated_rewards += rewards.astype(np.float32)
            if np.any(dones):
                break  # stop repeating if any env is done
        rewards = accumulated_rewards

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

        active_mask = np.ones(num_envs, dtype=np.float64)
        for di in envs.dead_envs:
            active_mask[di] = 0.0
            episode_returns[di] = 0.0
        episode_returns += rewards.astype(np.float64) * active_mask
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
                ep_ret = float(episode_returns[i])
                stat_entry = {
                    "episode_return": ep_ret,
                    "episode_length": int(step_ep),
                    "step": int(local_step),
                    "timestamp": time.time() - begin_time,
                    "train_test": train_test,
                    "type_mode": type_mode,
                    # action-smoothness metrics from wrapper (0.0 if absent)
                    "action_step_delta_mean": float(
                        info.get("action_step_delta_mean", 0.0)
                    ),
                    "action_step_delta_std": float(
                        info.get("action_step_delta_std", 0.0)
                    ),
                    "action_autocorr_lag1": float(
                        info.get("action_autocorr_lag1", 0.0)
                    ),
                }
                # forward Q-calibration data from test episodes to the learner
                if train_test == "test" and "q_calibration_data" in info:
                    stat_entry["q_calibration_data"] = info["q_calibration_data"]
                try:
                    stats_queue.put_nowait(stat_entry)
                except queue.Full:
                    print(
                        f"[ACTOR] stats_queue FULL — dropping episode {train_test} return={ep_ret:.3f} steps={step_ep}"
                    )
                    pass
                # DEBUG: print episode completion info
                print(
                    f"[ACTOR] env{i} {train_test.upper()} | type={type_mode} | return={ep_ret:.3f} | steps={step_ep}"
                )
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


def run_random_policy(d_arg):
    """Baseline: run a uniformly random policy, log episode stats, no learning."""
    device_str = (
        "cuda" if d_arg["simulation"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"[random] Running random policy baseline on {device_str}")

    envs = vec_envs(d_arg)
    obs = envs.reset()

    num_envs = envs.num_envs
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    total_steps = d_arg["rl"]["total_timesteps"]
    local_step = 0

    # Sliding-window return trackers for plots
    return_buffers = {
        "train": deque(maxlen=50),
        "test": deque(maxlen=50),
    }

    output_dir = d_arg["model"]["output_dir"]
    writer = SummaryWriter(log_dir=output_dir)

    if d_arg["simulation"]["wandb_track"]:
        run = wandb.init(
            project=d_arg["wandb"]["project"] if "wandb" in d_arg else "SAC_ASYNC_TIP",
            name=Path(output_dir).name,
            config=d_arg,
        )

    pbar = tqdm(total=total_steps, dynamic_ncols=True)

    action_repeat = d_arg["rl"].get("action_repeat", 1)

    try:
        while local_step < total_steps:
            actions = np.array(
                [envs.action_space.sample() for _ in range(num_envs)],
                dtype=np.float32,
            )

            accumulated_rewards = np.zeros(num_envs, dtype=np.float32)
            for _rep in range(action_repeat):
                next_obs, rewards, dones, infos = envs.step(actions)
                accumulated_rewards += rewards.astype(np.float32)
                if np.any(dones):
                    break
            rewards = accumulated_rewards

            active_mask = np.ones(num_envs, dtype=np.float64)
            for di in envs.dead_envs:
                active_mask[di] = 0.0
                episode_returns[di] = 0.0
            episode_returns += rewards.astype(np.float64) * active_mask
            local_step += num_envs - len(envs.dead_envs)

            for i in range(num_envs):
                if i in envs.dead_envs:
                    continue
                info = infos[i]
                if dones[i]:
                    split = info.get("train_test", "train")
                    typemode = info.get("type_mode", "unknown")
                    ep_ret = float(episode_returns[i])
                    ep_len = int(info.get("step_episode", 0))

                    a_delta_mean = float(info.get("action_step_delta_mean", 0.0))
                    a_delta_std = float(info.get("action_step_delta_std", 0.0))
                    a_autocorr = float(info.get("action_autocorr_lag1", 0.0))

                    return_buffers[split].append(ep_ret)

                    log_dict = {
                        f"charts/{split}_return_raw": ep_ret,
                        f"charts/{split}_{typemode}_return_raw": ep_ret,
                        f"charts/{split}_episode_length": ep_len,
                        f"charts/{split}_action_delta_mean": a_delta_mean,
                        f"charts/{split}_action_delta_std": a_delta_std,
                        f"charts/{split}_action_autocorr_lag1": a_autocorr,
                        f"charts/{split}_{typemode}_action_delta_mean": a_delta_mean,
                        f"charts/{split}_{typemode}_action_autocorr": a_autocorr,
                    }

                    buf = return_buffers[split]
                    if len(buf) >= 25:
                        log_dict[f"charts/{split}_return_mean"] = np.mean(buf)
                        log_dict[f"charts/{split}_return_std"] = np.std(buf)

                    if d_arg["simulation"]["wandb_track"]:
                        run.log(log_dict, step=local_step)
                    else:
                        for tag, val in log_dict.items():
                            writer.add_scalar(tag, val, local_step)
                    print(
                        f"[random] step={local_step}  ep_return={ep_ret:.3f}  "
                        f"ep_len={ep_len}  autocorr={a_autocorr:.3f}  mode={split}/{typemode}"
                    )
                    episode_returns[i] = 0.0

            obs = next_obs
            pbar.update(num_envs - len(envs.dead_envs))

    except KeyboardInterrupt:
        print("[random] Interrupted.")
    finally:
        pbar.close()
        envs.close()
        writer.close()
        if d_arg["simulation"]["wandb_track"]:
            wandb.finish()


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
    stats_queue = mp.Queue(maxsize=500)  # increased from 100 to handle episode bursts
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

    # ── Optional: resume from checkpoint ───────────────────────
    # All modules and optimizers are now materialized (dummy forward done,
    # optimizers built), so loading state dicts will not collide with
    # LazyLinear initialization.
    resume_grad_steps = 0
    resume_path = d_arg["rl"].get("resume_path", None)
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume path does not exist: {resume_path}")
        print(f"[checkpoint] resuming from {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=False)

        actor.load_state_dict(state["actor"])
        qf1.load_state_dict(state["qf1"])
        qf2.load_state_dict(state["qf2"])
        qf1_target.load_state_dict(state["qf1_target"])
        qf2_target.load_state_dict(state["qf2_target"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        q_optimizer.load_state_dict(state["q_optimizer"])

        if d_arg["rl"]["autotune"] and "log_alpha" in state:
            with torch.no_grad():
                log_alpha.copy_(state["log_alpha"].to(device))
            alpha_optim.load_state_dict(state["alpha_optim"])
            alpha = log_alpha.exp().item()

        resume_grad_steps = int(state.get("grad_steps", 0))
        print(
            f"[checkpoint] resumed at grad_steps={resume_grad_steps}, "
            f"drained={state.get('drained', 0)}"
        )

    # send initial policy to actor
    try:
        actor_queue.put_nowait(
            {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        )
    except queue.Full:
        actor_queue.put({k: v.detach().cpu() for k, v in actor.state_dict().items()})

    output_dir = d_arg["model"]["output_dir"]
    writer = SummaryWriter(log_dir=output_dir)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_frequency = d_arg["rl"].get("checkpoint_frequency", 10_000)

    def save_checkpoint(tag: str, step: int):
        path = os.path.join(ckpt_dir, f"sac_{tag}.pt")
        state = {
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            "grad_steps": step,
            "drained": drained,
            "d_arg_env": d_arg_env,
            "config": d_arg,
        }
        if d_arg["rl"]["autotune"]:
            state["log_alpha"] = log_alpha.detach().cpu()
            state["alpha_optim"] = alpha_optim.state_dict()
        torch.save(state, path)
        return path

    if d_arg["simulation"]["wandb_track"]:
        project_name = (
            d_arg["wandb"]["project"] if "wandb" in d_arg else "SAC_ASYNC_TIP"
        )
        print(f"[W&B INIT] project={project_name} | name={Path(output_dir).name}")
        try:
            run = wandb.init(
                project=project_name,
                name=Path(output_dir).name,
                config=d_arg,
            )
            # Define X-axis for all charts/* metrics
            run.define_metric("charts/*", step_metric="samples_drained")
            print(f"[W&B INIT] SUCCESS — run_id={run.id}")
        except Exception as e:
            print(f"[W&B INIT] FAILED: {e}")
            raise

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
        print(
            f"[CONFIG] total_timesteps={total_timesteps} learning_starts={learning_starts} "
            f"batch_size={batch_size} freq_test={d_arg['wrapper']['frequence_episode_test']}"
        )
        pbar = tqdm(total=total_timesteps, dynamic_ncols=True)
        grad_steps = resume_grad_steps
        _prev_drained = 0
        _grad_budget = 0.0
        _episode_count = {}

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

            # DEBUG: periodic summary every 50k samples
            if drained > 0 and drained % 50_000 == 0 and drained != _prev_drained:
                n_train = len(return_buffers["train"])
                n_test = len(return_buffers["test"])
                print(
                    f"\n[DEBUG] drained={drained:6d} | train_eps={n_train:3d} | test_eps={n_test:3d} | "
                    f"sample_rate={(drained - _prev_drained) / 50_000:.2f}x"
                )

            # ── PRIORITY: Drain stats_queue completely before any other computation ──
            # This ensures episode stats are processed ASAP, not blocked by GPU compute
            stats_processed = 0
            while not stats_queue.empty():
                try:
                    stat = stats_queue.get_nowait()
                except queue.Empty:
                    break
                stats_processed += 1

                split = stat["train_test"]
                typemode = stat.get("type_mode", "unknown")
                ep_return = stat["episode_return"]
                ep_length = stat.get("episode_length", 0)

                return_buffers[split].append(ep_return)

                # DEBUG: print episode completion info
                print(
                    f"[EPISODE] {split.upper()} | type={typemode} | return={ep_return:.3f} | steps={ep_length} | "
                    f"buf_size={len(return_buffers[split])} | drained={drained}"
                )

                a_delta_mean = stat.get("action_step_delta_mean", 0.0)
                a_delta_std = stat.get("action_step_delta_std", 0.0)
                a_autocorr = stat.get("action_autocorr_lag1", 0.0)

                # Primary metrics for paper plots
                log_dict = {
                    "samples_drained": drained,  # Track environment steps (visible in W&B)
                }

                # Rolling statistics (MAIN PLOT for paper: X=samples_drained, Y=return_mean)
                buf = return_buffers[split]
                if len(buf) >= 3:
                    log_dict[f"charts/{split}_return_mean"] = np.mean(buf)
                    log_dict[f"charts/{split}_return_std"] = np.std(buf)

                # Raw episode return (optional, for reference)
                log_dict[f"charts/{split}_return_raw"] = ep_return

                # Supporting metrics
                log_dict[f"charts/{split}_episode_length"] = (
                    ep_length  # how many steps per episode
                )

                # Action smoothness metrics
                log_dict[f"charts/{split}_action_delta_mean"] = a_delta_mean
                log_dict[f"charts/{split}_action_delta_std"] = a_delta_std
                log_dict[f"charts/{split}_action_autocorr_lag1"] = a_autocorr

                # ── Q-value calibration (test episodes only) ─────────
                q_calib = stat.get("q_calibration_data", None)
                if q_calib is not None and split == "test":
                    gamma = d_arg["rl"]["gamma"]
                    rewards_ep = np.array(
                        [s["reward"] for s in q_calib], dtype=np.float32
                    )
                    T = len(rewards_ep)
                    # discounted MC return from each step t
                    mc_returns = np.zeros(T, dtype=np.float32)
                    running = 0.0
                    for t in reversed(range(T)):
                        running = rewards_ep[t] + gamma * running
                        mc_returns[t] = running

                    is_graph = d_arg_env["is_graph"]
                    q_errors = []
                    qf1.eval()
                    qf2.eval()
                    with torch.no_grad():
                        for t, step_data in enumerate(q_calib):
                            obs_t = step_data["obs"]
                            act_t = step_data["action"]
                            if is_graph:
                                obs_t = {
                                    k: np.expand_dims(v, 0) for k, v in obs_t.items()
                                }
                                s = obs_to_pyg(obs_t, device)
                            else:
                                s = torch.tensor(
                                    obs_t, dtype=torch.float32, device=device
                                ).unsqueeze(0)
                            a = torch.tensor(
                                act_t, dtype=torch.float32, device=device
                            ).unsqueeze(0)
                            q_pred = torch.min(qf1(s, a), qf2(s, a)).item()
                            q_errors.append(q_pred - float(mc_returns[t]))
                    qf1.train()
                    qf2.train()

                    q_errors = np.array(q_errors)
                    log_dict["charts/test_q_bias"] = float(
                        np.mean(q_errors)
                    )  # + = overestimate
                    log_dict["charts/test_q_mae"] = float(np.mean(np.abs(q_errors)))
                    log_dict["charts/test_q_corr"] = (
                        float(np.corrcoef(mc_returns, mc_returns + q_errors)[0, 1])
                        if T > 1
                        else 0.0
                    )

                if d_arg["simulation"]["wandb_track"]:
                    run.log(log_dict, step=drained)
                    print(f"[W&B] Logged {len(log_dict)} metrics at drained={drained}")
                else:
                    for tag, value in log_dict.items():
                        writer.add_scalar(tag, value, drained)
                    print(f"[TB] Logged {len(log_dict)} metrics at drained={drained}")

            # ── Wait for enough samples ──────────────────────────
            if drained < max(learning_starts, batch_size):
                time.sleep(0.01)
                continue

            # ── Pace grad steps to drained (UTD = grad_utd) ──────
            # Accumulate budget so the outer loop spinning faster than
            # the drain thread never loses fractional credits.
            grad_utd = d_arg["rl"].get("grad_utd", 2.0)
            _grad_budget += (drained - _prev_drained) * grad_utd
            _prev_drained = drained
            if _grad_budget < 1.0:
                time.sleep(0.001)
                continue
            n_updates = min(int(_grad_budget), d_arg["rl"]["num_loops"])
            _grad_budget -= n_updates

            # ── Gradient updates ─────────────────────────────────
            for _ in range(n_updates):
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

                if grad_steps > 0 and grad_steps % ckpt_frequency == 0:
                    path = save_checkpoint(f"step{grad_steps:08d}", grad_steps)
                    print(f"[checkpoint] saved {path}")

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
        try:
            path = save_checkpoint("final", grad_steps)
            print(f"[checkpoint] final weights saved to {path}")
        except Exception as e:
            print(f"[checkpoint] failed to save final weights: {e}")

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
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--gpu", type=str, default="true")
    parser.add_argument("--observation_mode", default="transformer_nodes")
    parser.add_argument("--neural_architecture_image", default="impala")
    parser.add_argument("--max_time_episode", type=float, default=7200.0)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--total_timesteps", type=int, default=int(5e5))
    parser.add_argument("--rl_threads", type=int, default=4)
    parser.add_argument("--num_envs", type=int, default=13)
    parser.add_argument("--buffer_size", type=int, default=int(2e5))
    parser.add_argument("--batch_size_multiplier", type=int, default=64)
    parser.add_argument("--num_loops", type=int, default=3)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--checkpoint_frequency", type=int, default=50_000)
    parser.add_argument(
        "--grad_utd",
        type=float,
        default=1.0,
        help="Gradient steps per new env step (update-to-data ratio). "
        "1.0 = grads track drained. Same for all obs modes.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .pt checkpoint produced by save_checkpoint() to resume from.",
    )
    parser.add_argument("--name", default="TME_V2")
    parser.add_argument("--wandb", default="true")
    parser.add_argument("--entity", default="corporate-manu-sureli")
    parser.add_argument("--tumor", type=int, default=128)
    parser.add_argument("--Macrophage", type=int, default=32)
    parser.add_argument("--T_cells", type=int, default=32)
    parser.add_argument("--frequence_episode_test", type=int, default=4)
    parser.add_argument("--img_mc_grid_size", type=int, default=64)
    parser.add_argument("--w_cell", type=float, default=0.3)
    parser.add_argument(
        "--w_dose",
        type=float,
        default=1.0,
        help="Weight for dose penalty in reward function.",
    )
    parser.add_argument(
        "--w_smooth",
        type=float,
        default=0.0,
        help="Weight for action-smoothness penalty: reward -= w_smooth * ||a_t - a_{t-1}||^2",
    )
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=1,
        help="Number of steps the same action is repeated (frame skip). Default 1 = no repeat.",
    )
    parser.add_argument(
        "--delta_dose",
        type=float,
        default=None,
        help="Max change in dose per step (normalised [0,1]). None = unconstrained.",
    )
    parser.add_argument(
        "--delta_x",
        type=float,
        default=None,
        help="Max change in x per step (normalised [0,1]). None = unconstrained.",
    )
    parser.add_argument(
        "--delta_y",
        type=float,
        default=None,
        help="Max change in y per step (normalised [0,1]). None = unconstrained.",
    )
    parser.add_argument(
        "--delta_radius",
        type=float,
        default=None,
        help="Max change in radius per step (normalised [0,1]). None = unconstrained.",
    )
    parser.add_argument("--action_mode", type=str, default="full")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "random"],
        help="'train' runs SAC; 'random' runs a random-policy baseline.",
    )

    args = parser.parse_args()

    # ── Best hyperparameters from reward_analysis ──────────────────────────
    # Recommended: w_cell=0.3, w_dose=2.0, w_smooth=0.0 (composite_score=2.972)
    # These are the optimal hyperparameters found via offline reward sweeping.

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
        "project": "SAC_ASYNC_TME_NEW_HYP_REWARD",
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
        "action_mode": args.action_mode,
    }

    _var_names = (
        ["drug_1_dose"]
        if args.action_mode == "full"
        else ["drug_1_dose", "drug_1_x", "drug_1_y", "drug_1_radius"]
    )

    # Build per-component delta-max array only when action_mode == "targeted"
    # and at least one spatial delta limit is specified.
    # Order matches _var_names: [dose, x, y, radius].
    # Components left as None default to unconstrained (1.0 = full range).
    if args.action_mode == "targeted" and any(
        v is not None
        for v in [args.delta_dose, args.delta_x, args.delta_y, args.delta_radius]
    ):
        _action_delta_max = [
            args.delta_dose if args.delta_dose is not None else 1.0,
            args.delta_x if args.delta_x is not None else 1.0,
            args.delta_y if args.delta_y is not None else 1.0,
            args.delta_radius if args.delta_radius is not None else 1.0,
        ]
    else:
        _action_delta_max = None

    d_arg_physigym_wrapper = {
        "list_variable_name": _var_names,
        "w_cell": args.w_cell,
        "w_dose": args.w_dose,
        "w_smooth": args.w_smooth,
        "action_delta_max": _action_delta_max,
        "frequence_episode_test": args.frequence_episode_test,
        "action_mode": args.action_mode,
    }

    d_arg_rl = {
        "total_timesteps": args.total_timesteps,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size_multiplier * args.num_envs,
        "learning_starts": args.learning_starts,
        "num_loops": args.num_loops,
        "policy_frequency": args.policy_frequency,
        "target_network_frequency": args.target_network_frequency,
        "checkpoint_frequency": args.checkpoint_frequency,
        "grad_utd": args.grad_utd,
        "action_repeat": args.action_repeat,
        "resume_path": args.resume,
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
        "mode_train": ["rectangle"],
        "mode_test": ["network_field"],
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
        f"{d_arg['wrapper']['action_mode']}_"
        f"{int(time.time())}"
    )

    if args.mode == "random":
        run_random_policy(d_arg=d_arg)
    else:
        run_async_sac(d_arg=d_arg)
