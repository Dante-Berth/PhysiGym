"""Replay test episodes from a saved SAC checkpoint and compare Q with three yardsticks.

Purpose (Ch. 6, value section): decide whether the downward drift of the logged
calibration error b = Q - G_MC under uniform action is the entropy term of the soft
value.  For each episode the script records the quantities needed for:

  logged   the convention of run.py / wrapper.py: one row per SIMULATOR step, Q(next_obs,
           a_held), reward per sim step, gamma applied per sim step.  Reproduces test_q_bias.
  dec_plain  the convention the critic is TRAINED on: one row per DECISION (block of
           action_repeat sim steps), block-summed reward, gamma per decision, Q(obs_at_block_start, a).
  dec_soft   dec_plain with the entropy term:  G_soft_k = R_k + sum_{j>=1} gamma^j (R_{k+j} - alpha*log pi_{k+j}).

If the entropy explanation holds, |b_dec_soft| << |b_dec_plain| ~ |b_logged|.  Actions are
SAMPLED from the policy (as in the training loop's test episodes).

Run from the vroom repo root:
    cd ~/PhysiCell_vroom_vroom
    PYTHONPATH=custom_modules/physigym/physigym/envs .venv/bin/python soft_q_eval.py \
        --ckpt <path.pt> --episodes 8 --out <dir> --tag <label>
Writes <out>/<tag>.json (one record per episode).  Training data is never touched.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

torch.set_num_threads(1)


def build(ckpt_path, ic_seed, tmp_dir):
    from vectorized import make_physigym_env
    from nn import Actor, QNetwork

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cfg["vectorization"]["num_envs"] = 1
    cfg["vectorization"]["rl_threads"] = 1
    cfg["vectorization"]["threads_per_env"] = 1
    cfg["simulation"]["wandb_track"] = False
    cfg["model"]["render_mode"] = None
    test_family = cfg["generation"]["mode_test"]
    cfg["generation"]["mode_train"] = list(test_family)      # every episode from the HELD-OUT family
    cfg["generation"]["seed"] = ic_seed
    cfg["wrapper"]["frequence_episode_test"] = 1             # every episode is a test episode
    cfg["model"]["output_dir"] = tmp_dir
    env = make_physigym_env(0, cfg)()
    d_env = ck["d_arg_env"]
    actor = Actor(d_env, cfg["neural_architecture_image"])
    q1 = QNetwork(d_env, cfg["neural_architecture_image"])
    q2 = QNetwork(d_env, cfg["neural_architecture_image"])
    dummy = torch.zeros((1, *d_env["observation_space_shape"]), dtype=torch.float32)
    with torch.no_grad():
        a0, _, _ = actor.get_action(dummy)
        q1(dummy, a0); q2(dummy, a0)                         # materialise lazy layers
    actor.load_state_dict(ck["actor"]); q1.load_state_dict(ck["qf1"]); q2.load_state_dict(ck["qf2"])
    for m in (actor, q1, q2):
        m.eval()
    alpha = float(ck["log_alpha"].exp().item())
    return env, actor, q1, q2, alpha, cfg, ck


def discounted(rew, gamma, extra=None):
    """G_k = rew_k + gamma G_{k+1}; with extra: G_k = rew_k + sum_{j>=1} gamma^j (rew_{k+j} + extra_{k+j})."""
    T = len(rew); out = np.zeros(T)
    run = 0.0
    for k in reversed(range(T)):
        step = rew[k]
        out[k] = step + gamma * run
        run = (rew[k] + (extra[k] if extra is not None else 0.0)) + gamma * run
    return out


def episode(env, actor, q1, q2, alpha, gamma, AR):
    obs, _ = env.reset()
    obs_np = np.asarray(obs)
    sim_obs, sim_act, sim_r = [], [], []
    dec_obs, dec_act, dec_R, dec_logp = [], [], [], []
    done = False
    while not done:
        x = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a, logp, _ = actor.get_action(x)
        a_np = a.squeeze(0).numpy().astype(np.float32)
        dec_obs.append(obs_np.copy()); dec_act.append(a_np); dec_logp.append(float(logp.item()))
        R = 0.0
        for _ in range(AR):
            nobs, r, term, trunc, info = env.step(a_np)
            sim_obs.append(np.asarray(nobs).copy()); sim_act.append(a_np); sim_r.append(float(r))
            R += float(r)
            obs_np = np.asarray(nobs)
            if term or trunc:
                done = True
                break
        dec_R.append(R)
    def qmin(obs_list, act_list):
        out = []
        with torch.no_grad():
            for i in range(0, len(obs_list), 64):
                s = torch.as_tensor(np.stack(obs_list[i:i + 64]), dtype=torch.float32)
                aa = torch.as_tensor(np.stack(act_list[i:i + 64]), dtype=torch.float32)
                out.append(torch.min(q1(s, aa), q2(s, aa)).squeeze(-1).numpy())
        return np.concatenate(out)
    # logged convention (sim steps)
    Qs = qmin(sim_obs, sim_act); Gs = discounted(np.array(sim_r), gamma)
    # decision convention
    Qd = qmin(dec_obs, dec_act); R = np.array(dec_R); lp = np.array(dec_logp)
    Gd = discounted(R, gamma)
    Gsoft = discounted(R, gamma, extra=-alpha * lp)
    # Time-limit correction: episodes cut at the horizon (80 decisions) are not terminal, the critic
    # never sees the elapsed time and bootstraps as if the episode went on.  Add the discounted
    # continuation of the soft reward, estimated from the mean of the last 10 decisions.
    T = len(R); truncated = T >= 80
    tail = float(np.mean(R[-10:] - alpha * lp[-10:])) if truncated else 0.0
    ext = np.array([gamma ** (T - k) for k in range(T)]) * tail / (1.0 - gamma)
    Gsoft_ext = Gsoft + ext
    return dict(
        b_dec_soft_ext=float(np.mean(Qd - Gsoft_ext)), ext_term=float(np.mean(ext)), tail_soft_reward=tail,
        truncated=bool(truncated),
        T_sim=len(sim_r), T_dec=len(dec_R), alpha=alpha,
        b_logged=float(np.mean(Qs - Gs)), m_logged=float(np.mean(np.abs(Qs - Gs))),
        b_dec_plain=float(np.mean(Qd - Gd)), m_dec_plain=float(np.mean(np.abs(Qd - Gd))),
        b_dec_soft=float(np.mean(Qd - Gsoft)), m_dec_soft=float(np.mean(np.abs(Qd - Gsoft))),
        ent_term=float(np.mean(Gd - Gsoft)),                  # = b_dec_plain - b_dec_soft
        mean_logp=float(lp.mean()), mean_Q=float(Qd.mean()), mean_Gd=float(Gd.mean()),
        mean_Gsoft=float(Gsoft.mean()), mean_Gs=float(Gs.mean()), mean_Qs=float(Qs.mean()),
        ep_return=float(np.sum(sim_r)),
        mean_dose=float(np.mean([a[0] for a in dec_act])),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--ic_seed", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp_" + a.tag)
    env, actor, q1, q2, alpha, cfg, ck = build(a.ckpt, a.ic_seed, tmp)
    gamma = float(cfg["rl"]["gamma"]); AR = int(cfg["rl"].get("action_repeat", 1))
    jpath = os.path.join(a.out, a.tag + ".json")
    rec = json.load(open(jpath)) if os.path.exists(jpath) else []   # resume after a worker crash
    t0 = time.time()
    for e in range(len(rec), a.episodes):
        r = episode(env, actor, q1, q2, alpha, gamma, AR)
        r.update(tag=a.tag, ckpt=os.path.basename(a.ckpt), episode=e, gamma=gamma, AR=AR,
                 grad_steps=int(ck.get("grad_steps", -1)), seconds=time.time() - t0)
        rec.append(r)
        print(f"[{a.tag}] ep{e} T_dec={r['T_dec']} b_logged={r['b_logged']:+.1f} "
              f"b_dec_plain={r['b_dec_plain']:+.1f} b_dec_soft={r['b_dec_soft']:+.1f} "
              f"ent={r['ent_term']:+.1f} logp={r['mean_logp']:+.2f} alpha={alpha:.2f}", flush=True)
        json.dump(rec, open(os.path.join(a.out, a.tag + ".json"), "w"))
    env.close()
