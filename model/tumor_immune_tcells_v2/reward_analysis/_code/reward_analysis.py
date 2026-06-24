#####
# title: reward_analysis.py
#
# Quantitative analysis of the PhysiCell drug-dosing reward function across
# reward-shaping hyperparameters.
#
# Goal of the study
# ─────────────────
# We want a reward that produces, at the end of training, a SMOOTH dosing
# policy where:
#   (1) adding drug is PENALISED   (dose is a cost, not free),
#   (2) killing cancer is ENCOURAGED,
#   (3) but the killing is mechanistically done by T CELLS — the drug only
#       reshapes the micro-environment — so reward should CREDIT the
#       T-cell-mediated kill and NOT reward "blind dumping" of drug,
#   (4) and image observations should be able to OUTPERFORM scalar
#       observations (spatial targeting only an image policy can perceive).
#
# Method
# ──────
# The reward weights do not change the physics. So we run each
# (observation_mode, seed) ONCE with a fixed reference policy, log every raw
# reward component per step (dose_spent, r_cancer_cells, tumor_killed,
# n_tcell, smooth_penalty, action_*), then RECOMPUTE the shaped reward for a
# whole grid of hyperparameters analytically over those logged trajectories.
# This makes a dense sweep cheap: O(n_seeds * n_modes) simulations, not
# O(grid_size) simulations.
#
#   shaped_reward = w_cell  * r_cancer_cells   (encourage killing)
#                 - w_dose  * dose_spent       (penalise drug; w_dose=1 == env)
#                 - w_smooth* smooth_penalty   (encourage smoothness)
#
# This is exactly the reward in wrapper_tip.py, with the (normally fixed = 1)
# dose coefficient exposed as a sweepable knob. T-cell / macrophage counts are
# LOGGED and PLOTTED for mechanism attribution (the drug doesn't kill — T cells
# do) but are NOT part of the reward.
#
# Outputs (written to --outdir):
#   - reward_sweep_ranking.csv   full ranking table over the hyperparameter grid
#   - reward_sweep_report.md     human-readable summary + recommendation
#   - fig_dose_vs_kill.png       dose-penalty effectiveness
#   - fig_smoothness.png         smoothness vs w_smooth
#   - fig_img_vs_scalars.png     img - scalars reward gap per config
#   - fig_attribution.png        T-cell-attributed vs raw kill
#####

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────
# 1. Rollout: run each (observation_mode, seed) once, log raw components
# ──────────────────────────────────────────────────────────────────────────

# Raw per-step columns the rollout must produce. These are physics-only and
# independent of reward weights, so the sweep can recompute reward over them.
RAW_COLUMNS = [
    "dyn", "policy", "obs_mode", "seed", "episode", "step",
    "r_cancer_cells", "dose_spent", "tumor_killed",
    "n_tcell", "n_macrophage", "number_tumor",
    "action_dose", "action_x", "action_y", "action_radius",
    "smooth_penalty",
]


# ── reference policies (weight-independent action generators) ──────────────
# These three baselines are the core diagnostic: a good reward must rank them
# sensibly. The drug does NOT kill tumor (T cells do), so "zero_drug" should
# not be punished relative to "max_drug_fixed", and blind dumping must not win.
#   random          : random dose + random target each step  (explores space)
#   zero_drug        : dose = 0 always                        (do-nothing baseline)
#   max_drug_fixed   : dose = 1 at a fixed centre position    (always-dump baseline)

_FIXED_POS = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # x, y, radius (normalised)


# cosine ("pulsed therapy") period, in decision steps — one full 0→1→0 cycle
_COS_PERIOD = 20


def policy_action(name, rng, prev_raw, t=0):
    if name == "random":
        return rng.uniform(0.0, 1.0, size=4).astype(np.float32)
    if name == "zero_drug":
        return np.concatenate([[0.0], _FIXED_POS]).astype(np.float32)
    if name == "max_drug_fixed":
        return np.concatenate([[1.0], _FIXED_POS]).astype(np.float32)
    if name == "cosine":
        # pulsed regimen: dose oscillates smoothly in [0,1] at a fixed centre.
        # dose = (1 - cos(2πt/T)) / 2  → starts at 0, peaks at T/2, back to 0.
        dose = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / _COS_PERIOD))
        return np.concatenate([[dose], _FIXED_POS]).astype(np.float32)
    raise ValueError(f"unknown policy {name}")


POLICIES = ["random", "zero_drug", "max_drug_fixed", "cosine"]


def run_rollouts(obs_modes, seeds, episodes_per_seed, max_time, base_xml, base_cells,
                 policies=POLICIES):
    """
    Run fresh PhysiCell episodes via the vectorized_tip factory and collect raw
    per-step reward components for each reference policy. Returns a long-form
    DataFrame (RAW_COLUMNS).

    For every (obs_mode, seed) we run each policy on the SAME initial conditions
    (same seed sequence) so policies are compared on matched tumours.
    """
    # imported here so the file is importable for --replot without a built sim
    from vectorized_tip import make_physigym_env

    rows = []
    for obs_mode in obs_modes:
        for seed in seeds:
            cfg = _build_cfg(obs_mode, seed, max_time, base_xml, base_cells)
            print(f"[rollout] obs_mode={obs_mode} seed={seed} ...", flush=True)
            env = make_physigym_env(0, cfg)()
            try:
                for policy in policies:
                    # reseed RNG per policy so 'random' is reproducible & comparable
                    rng = np.random.default_rng(seed * 1000 + hash(policy) % 997)
                    for ep in range(episodes_per_seed):
                        env.reset()
                        env.generate_physicell_data = False  # no video; raw logs only
                        prev_raw = None
                        done = False
                        t = 0
                        while not done:
                            a = policy_action(policy, rng, prev_raw, t=t)
                            obs, reward, terminated, truncated, info = env.step(a)
                            prev_raw = a
                            t += 1
                            done = terminated or truncated
                        for r in env.list_data:
                            rows.append({
                                "dyn":            "recommended",
                                "policy":         policy,
                                "obs_mode":       obs_mode,
                                "seed":           seed,
                                "episode":        ep,
                                "step":           r.get("step", 0),
                                "r_cancer_cells": r.get("r_cancer_cells", r.get("reward", 0.0)),
                                "dose_spent":     r.get("dose_spent", 0.0),
                                "tumor_killed":   r.get("tumor_killed", 0.0),
                                "n_tcell":        r.get("n_tcell", 0),
                                "n_macrophage":   r.get("n_macrophage", 0),
                                "number_tumor":   r.get("number_tumor", 0),
                                "action_dose":    r.get("action_dose", 0.0),
                                "action_x":       r.get("action_x", 0.0),
                                "action_y":       r.get("action_y", 0.0),
                                "action_radius":  r.get("action_radius", 0.0),
                                "smooth_penalty": r.get("smooth_penalty", 0.0),
                            })
                        env.list_data = []
            finally:
                try:
                    env.close()
                except Exception:
                    pass

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


# ── dynamics hyperparameter configs ────────────────────────────────────────
# These change the TRAJECTORY physics (action repeat + per-step delta clipping)
# so, unlike reward weights, each must be measured with its own rollout.
# 'name' tags every row so the analysis can plot one curve per config.
DEFAULT_DYNAMICS = [
    # the recommended publication config (run.sh)
    dict(name="recommended", action_repeat=4, delta=[1.0, 0.15, 0.15, 0.05]),
    # no action repeat, same deltas — isolates the effect of holding actions
    dict(name="no_repeat",   action_repeat=1, delta=[1.0, 0.15, 0.15, 0.05]),
    # unconstrained (twitchy) — no delta clip, no repeat: the naive baseline
    dict(name="unconstrained", action_repeat=1, delta=None),
    # smoother/slower targeting: longer repeat + tighter spatial deltas
    dict(name="slow_smooth", action_repeat=8, delta=[1.0, 0.08, 0.08, 0.03]),
]


def run_rollouts_parallel(obs_modes, seeds, episodes_per_seed, max_time,
                          base_xml, base_cells, policies=POLICIES, dyn=None):
    """
    Parallel version: one SubprocVecEnv worker per seed, all stepped in lockstep.

    For each observation mode we spin up len(seeds) envs (each a distinct seed)
    and run every policy until each env has logged `episodes_per_seed` finished
    episodes. Per-step rows are harvested from info['reward_analysis_rows'],
    which the wrapper attaches at episode end (survives the subprocess
    auto-reset that wipes list_data).
    """
    from vectorized_tip import vec_envs

    if dyn is None:
        dyn = DEFAULT_DYNAMICS

    rows = []
    for dcfg in dyn:
        for obs_mode in obs_modes:
            # vec_envs derives each worker's seed from a single master seed; we
            # use the FIRST requested seed as master and label rows by worker
            # index so each parallel env is a distinct, reproducible tumour.
            master = seeds[0]
            cfg = _build_cfg(obs_mode, master, max_time, base_xml, base_cells)
            cfg["simulation"]["seed"] = master
            cfg["vectorization"]["num_envs"] = len(seeds)
            # apply this dynamics config to the wrapper delta-clip
            cfg["wrapper"]["action_delta_max"] = dcfg["delta"]
            action_repeat = int(dcfg.get("action_repeat", 1))
            env_labels = [master * 100 + i for i in range(len(seeds))]
            print(f"[rollout|| ] dyn={dcfg['name']:>13}  obs_mode={obs_mode}  "
                  f"{len(seeds)} envs || (repeat={action_repeat}, delta={dcfg['delta']})",
                  flush=True)

            envs = vec_envs(cfg)
            envs.set_attr("emit_reward_analysis_rows", True)
            envs.set_attr("generate_physicell_data", False)
            n_envs = envs.num_envs
            try:
                for policy in policies:
                    envs.reset()
                    rng = np.random.default_rng(1234 + hash(policy) % 997)
                    done_counts = np.zeros(n_envs, dtype=int)
                    prev = [None] * n_envs
                    t_env = np.zeros(n_envs, dtype=int)  # per-env decision index (for cosine)
                    while (done_counts < episodes_per_seed).any():
                        # one DECISION: pick action, hold it for action_repeat steps
                        acts = np.stack([
                            policy_action(policy, rng, prev[i], t=int(t_env[i]))
                            for i in range(n_envs)
                        ]).astype(np.float32)
                        prev = [acts[i] for i in range(n_envs)]
                        t_env += 1
                        for _rep in range(action_repeat):
                            _, _, dones, infos = envs.step(acts)
                            for i, (d, info) in enumerate(zip(dones, infos)):
                                if d and done_counts[i] < episodes_per_seed:
                                    for r in info.get("reward_analysis_rows", []):
                                        rows.append(_raw_row(
                                            r, policy, obs_mode,
                                            env_labels[i], int(done_counts[i]),
                                            dyn_name=dcfg["name"]))
                                    done_counts[i] += 1
                                    t_env[i] = 0  # restart cosine cycle for next episode
                            if (done_counts >= episodes_per_seed).all():
                                break
            finally:
                try:
                    envs.close()
                except Exception:
                    pass

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _raw_row(r, policy, obs_mode, seed, episode, dyn_name="recommended"):
    return {
        "dyn": dyn_name,
        "policy": policy, "obs_mode": obs_mode, "seed": seed, "episode": episode,
        "step":           r.get("step", 0),
        "r_cancer_cells": r.get("r_cancer_cells", r.get("reward", 0.0)),
        "dose_spent":     r.get("dose_spent", 0.0),
        "tumor_killed":   r.get("tumor_killed", 0.0),
        "n_tcell":        r.get("n_tcell", 0),
        "n_macrophage":   r.get("n_macrophage", 0),
        "number_tumor":   r.get("number_tumor", 0),
        "action_dose":    r.get("action_dose", 0.0),
        "action_x":       r.get("action_x", 0.0),
        "action_y":       r.get("action_y", 0.0),
        "action_radius":  r.get("action_radius", 0.0),
        "smooth_penalty": r.get("smooth_penalty", 0.0),
    }


def _build_cfg(obs_mode, seed, max_time, base_xml, base_cells):
    # MUST mirror run.py so the analysis measures the env the policy trains on:
    #   tumor=128, macrophage=32, t_cell=32; normalization_factor=128;
    #   max_time_episode=7200; mode_train/test=["rectangle"]; action_mode=targeted.
    params = {
        "tumor":      {"correlation_length": 45, "threshold": 0.55, "number_cells": 128},
        "macrophage": {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
        "t_cell":     {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
    }
    return {
        "simulation":    {"max_time": max_time, "seed": seed},
        "vectorization": {"num_envs": 1, "rl_threads": 1, "threads_per_env": 1},
        "model": {
            "id": "physigym/ModelPhysiCellEnv-v0",
            "settingxml": base_xml,
            "settingcells": base_cells,
            "output_dir": "./reward_analysis_output",
            "figsize": (6, 6),
            "observation_mode": obs_mode,
            "action_mode": "targeted",   # expose drug_1_x/y/radius (not just dose)
            "render_mode": None,
            "verbose": False,
            "img_rgb_grid_size_x": 64, "img_rgb_grid_size_y": 64,
            "img_mc_grid_size_x": 64, "img_mc_grid_size_y": 64,
            "normalization_factor": 128,   # = run.py --tumor
        },
        "wrapper": {
            "list_variable_name": ["drug_1_dose", "drug_1_x", "drug_1_y", "drug_1_radius"],
            "w_cell": 0.3, "w_smooth": 0.02,
            "action_mode": "targeted",
            "action_delta_max": [1.0, 0.15, 0.15, 0.05],
        },
        "generation": {
            "x_min": 0, "x_max": 64, "y_min": 0, "y_max": 64,
            "params": params, "seed": seed,
            # user-requested initial distribution: network_field for both splits
            "mode_train": ["network_field"],
            "mode_test":  ["network_field"],
        },
        "rl": {"total_timesteps": 25000},
    }


# ──────────────────────────────────────────────────────────────────────────
# 2. Reward recomputation over the hyperparameter grid
# ──────────────────────────────────────────────────────────────────────────

def tcell_attribution(df, tcell_ref=8.0):
    """
    Fraction of a step's tumor kill that is credibly T-cell-mediated.

    A kill happening with many T cells present is credited; a tumor decrease
    with no T cells around is attributed to raw drug toxicity and earns ~0.
    Smooth saturating gate in [0,1] = n_tcell / (n_tcell + tcell_ref).
    """
    n = df["n_tcell"].to_numpy(dtype=np.float64)
    gate = n / (n + tcell_ref)
    # only positive kills get credited; growth (negative) is unaffected by gate
    kill = df["tumor_killed"].to_numpy(dtype=np.float64)
    credited = np.where(kill > 0, kill * gate, kill)
    # normalise to the same scale as r_cancer_cells by reusing its sign/scale:
    # r_cancer_cells = kill / expected_growth, so attributed = r_cancer_cells * gate
    rcc = df["r_cancer_cells"].to_numpy(dtype=np.float64)
    return np.where(rcc > 0, rcc * gate, rcc)


def shaped_reward(df, w_cell, w_dose, w_smooth):
    """
    The real wrapper_tip.py reward, with the (normally fixed = 1) dose
    coefficient exposed as a sweepable knob. w_dose=1 reproduces the env exactly:
        reward = w_cell * r_cancer_cells - w_dose * dose_spent - w_smooth * smooth_penalty
    T-cell signals are NOT part of the reward; they are only used for the
    diagnostic attribution metric below.
    """
    return (
        w_cell  * df["r_cancer_cells"].to_numpy(dtype=np.float64)
        - w_dose  * df["dose_spent"].to_numpy(dtype=np.float64)
        - w_smooth * df["smooth_penalty"].to_numpy(dtype=np.float64)
    )


def episode_metrics(df, r):
    """Per-episode aggregate metrics for a recomputed reward array `r`.

    Grouped by policy too: each (policy, obs_mode, seed, episode) is a distinct
    trajectory. Omitting policy would collapse the three reference policies'
    episodes into one group and make their returns indistinguishable.
    """
    df = df.assign(_r=r)
    g = df.groupby(["dyn", "policy", "obs_mode", "seed", "episode"])
    out = g.agg(
        ep_return     =("_r", "sum"),
        total_dose    =("dose_spent", "sum"),
        total_kill    =("tumor_killed", "sum"),
        mean_tcell    =("n_tcell", "mean"),
        smooth_cost   =("smooth_penalty", "sum"),
        start_tumor   =("number_tumor", "first"),
        final_tumor   =("number_tumor", "last"),
        steps         =("_r", "size"),
    ).reset_index()
    # the three outcomes the policy must satisfy, in interpretable units:
    #   tumor_reduction : start − final cancer cells   (want > 0: cancer decreases)
    #   total_dose      : cumulative drug spent         (want small)
    #   smooth_cost     : summed action jitter          (want small)
    out["tumor_reduction"] = out["start_tumor"] - out["final_tumor"]
    return out


def evaluate_config(raw, w_cell, w_dose, w_smooth):
    """
    Compute the four study metrics for one hyperparameter configuration.

    Returns a dict of scalar scores (higher = better for each except where noted).
    """
    attr = tcell_attribution(raw)   # diagnostic only — not in the reward
    r = shaped_reward(raw, w_cell, w_dose, w_smooth)
    ep = episode_metrics(raw, r)

    # (1) dose penalty effectiveness: across episodes, return should ANTI-correlate
    # with dose given equal killing — i.e. high dose should not be rewarded.
    # Measure partial: corr(ep_return, total_dose) controlling for total_kill.
    dose_penalty_corr = _partial_corr(ep["ep_return"], ep["total_dose"], ep["total_kill"])

    # (2) kill encouragement: return should correlate POSITIVELY with kill.
    kill_reward_corr = _safe_corr(ep["ep_return"], ep["total_kill"])

    # (3) T-cell attribution: per STEP, reward credit should track T-cell presence
    # more than raw dose. Compare corr(r, attributed) vs corr(r, dose).
    step_attr_corr = _safe_corr(r, attr)
    step_dose_corr = _safe_corr(r, raw["dose_spent"].to_numpy(dtype=np.float64))
    attribution_margin = step_attr_corr - step_dose_corr  # want > 0

    # (4) img > scalars: mean episode return for img modes minus scalar modes.
    is_img = ep["obs_mode"].str.contains("img")
    img_ret    = ep.loc[is_img, "ep_return"].mean()    if is_img.any()    else np.nan
    scalar_ret = ep.loc[~is_img, "ep_return"].mean()   if (~is_img).any() else np.nan
    img_minus_scalar = (img_ret - scalar_ret) if np.isfinite(img_ret) and np.isfinite(scalar_ret) else np.nan

    # smoothness: lower per-episode smooth cost is better. Report mean.
    mean_smooth_cost = ep["smooth_cost"].mean()

    # policy ranking gaps (the user's headline concern)
    psum = policy_comparison(raw, w_cell, w_dose, w_smooth)
    gap_zero_vs_max   = psum.attrs.get("gap_zero_vs_max", np.nan)
    gap_random_vs_max = psum.attrs.get("gap_random_vs_max", np.nan)

    return {
        "w_cell": w_cell, "w_dose": w_dose, "w_smooth": w_smooth,
        "dose_penalty_corr":  dose_penalty_corr,    # want negative
        "kill_reward_corr":   kill_reward_corr,     # want positive
        "attribution_margin": attribution_margin,   # want positive
        "img_minus_scalar":   img_minus_scalar,     # want positive
        "mean_smooth_cost":   mean_smooth_cost,     # want small
        "gap_zero_vs_max":    gap_zero_vs_max,      # want >= 0 (don't pay for dumping)
        "gap_random_vs_max":  gap_random_vs_max,    # want > 0  (targeting can win)
        "mean_ep_return":     ep["ep_return"].mean(),
    }


def policy_comparison(raw, w_cell, w_dose, w_smooth):
    """
    Headline diagnostic the user asked for: under a given reward, what is the
    mean CUMULATIVE return of each fixed policy?

    Returns a tidy DataFrame: one row per (policy) with mean ± std episode
    return, plus the two decisive gaps:
      gap_zero_vs_max   = return(zero_drug)  − return(max_drug_fixed)
                          > 0  ⟹ reward correctly does NOT pay for blind dumping
      gap_random_vs_max = return(random)     − return(max_drug_fixed)
                          > 0  ⟹ exploring/targeting beats always-max-dose
    """
    r = shaped_reward(raw, w_cell, w_dose, w_smooth)
    ep = episode_metrics(raw, r)  # keyed by dyn, policy, obs_mode, seed, episode
    # per (dyn, policy) mean cumulative return
    summary = (ep.groupby(["dyn", "policy"])["ep_return"]
                 .agg(mean_return="mean", std_return="std", n="size")
                 .reset_index())
    # gaps computed per dynamics config (pooled gap = mean over dyn for ranking)
    gaps_zero, gaps_rand = [], []
    for d, sub in summary.groupby("dyn"):
        m = sub.set_index("policy")["mean_return"].to_dict()
        gaps_zero.append(m.get("zero_drug", np.nan) - m.get("max_drug_fixed", np.nan))
        gaps_rand.append(m.get("random", np.nan) - m.get("max_drug_fixed", np.nan))
    summary.attrs["gap_zero_vs_max"]   = float(np.nanmean(gaps_zero)) if gaps_zero else np.nan
    summary.attrs["gap_random_vs_max"] = float(np.nanmean(gaps_rand)) if gaps_rand else np.nan
    return summary


def plot_policy_comparison(raw, configs, outdir):
    """
    Bar chart of mean cumulative return per policy, one group per reward config.
    This directly answers: how do random vs 0-drug vs 1-drug-fixed compare?
    """
    labels = [f"wc{wc}_wd{wd}_ws{ws}" for (wc, wd, ws) in configs]
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(configs)), 5))
    width = 0.25
    pol_order = ["zero_drug", "random", "max_drug_fixed"]
    colors = {"zero_drug": "#264653", "random": "#2a9d8f", "max_drug_fixed": "#e76f51"}
    x = np.arange(len(configs))
    for i, pol in enumerate(pol_order):
        means, errs = [], []
        for cfg in configs:
            s = policy_comparison(raw, *cfg)
            row = s[s["policy"] == pol]
            means.append(float(row["mean_return"].iloc[0]) if len(row) else np.nan)
            errs.append(float(row["std_return"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3,
               label=pol, color=colors[pol])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean cumulative episode return")
    ax.set_title("Cumulative return by policy across reward configs\n"
                 "(want zero_drug ≥ max_drug_fixed; random able to beat max_drug_fixed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_policy_comparison.png"), dpi=130)
    plt.close(fig)


def outcome_tradeoff(raw, grid, policy="random"):
    """
    The view the user actually wants: for each reward weight config, the THREE
    real outcomes a good policy must achieve, in interpretable units.

    Evaluated on the `random` probe policy (it exercises the whole action range,
    so its outcomes respond to the reward weights the way a trained policy's
    incentives would). Returns one row per (w_cell, w_dose, w_smooth):
        tumor_reduction : mean (start − final) cancer cells   ↑ better
        total_dose      : mean cumulative drug spent          ↓ better
        smooth_cost     : mean summed action jitter           ↓ better
        mean_return     : mean shaped return under this reward
    """
    rows = []
    for (wc, wd, ws) in grid:
        r = shaped_reward(raw, wc, wd, ws)
        ep = episode_metrics(raw, r)
        ep = ep[ep["policy"] == policy]
        if ep.empty:
            continue
        # group by dynamics config — the trajectory (hence the 3 outcomes) varies
        # with dynamics, not with reward weights on a fixed probe policy.
        for dyn_name, sub in ep.groupby("dyn"):
            rows.append({
                "dyn": dyn_name,
                "w_cell": wc, "w_dose": wd, "w_smooth": ws,
                "tumor_reduction": sub["tumor_reduction"].mean(),
                "total_dose":      sub["total_dose"].mean(),
                "smooth_cost":     sub["smooth_cost"].mean(),
                "mean_return":     sub["ep_return"].mean(),
                "final_tumor":     sub["final_tumor"].mean(),
            })
    return pd.DataFrame(rows)


def plot_outcome_tradeoff(raw, grid, outdir, policy="random"):
    """
    Three-axis trade-off: tumor reduction (want high) vs total dose (want low),
    point colour = smoothness cost (want low). The ideal reward sits top-left in
    a cool colour. Annotates each point with its (w_cell, w_dose, w_smooth).
    """
    to = outcome_tradeoff(raw, grid, policy=policy)
    if to.empty:
        return to
    # outcomes vary with the dynamics config (trajectory), so one point per dyn
    # (deduplicate across reward weights, which don't move a fixed-policy outcome)
    pts = to.drop_duplicates(subset=["dyn"])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(pts["total_dose"], pts["tumor_reduction"],
                    c=pts["smooth_cost"], cmap="viridis_r", s=140, edgecolor="k")
    for _, row in pts.iterrows():
        ax.annotate(str(row["dyn"]),
                    (row["total_dose"], row["tumor_reduction"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("total drug dose  (← less drug is better)")
    ax.set_ylabel("tumor reduction  (more cancer killed is better →)")
    ax.set_title(f"Outcome trade-off across dynamics configs ('{policy}' policy)\n"
                 "ideal = top-left, dark = smoother")
    fig.colorbar(sc, ax=ax, label="action jitter / smoothness cost (lower better)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_outcome_tradeoff.png"), dpi=130)
    plt.close(fig)
    to.to_csv(os.path.join(outdir, "outcome_tradeoff.csv"), index=False)
    return to


_POLICY_COLORS = {
    "zero_drug": "#264653", "random": "#2a9d8f",
    "max_drug_fixed": "#e76f51", "cosine": "#9b5de5",
}


def plot_time_series(raw, w_cell, w_dose, w_smooth, outdir, dyn_name=None):
    """
    Per-step EVOLUTION over an episode, one line per policy:
      (1) cumulative reward     — how return accrues under this reward
      (2) cancer cell count     — does the tumour actually shrink?
      (3) dose applied / step   — the dynamic treatment regime (incl. cosine pulse)
      (4) cumulative dose       — total drug burden over time

    Curves are the per-step mean across seeds/episodes (shaded = ±1 std), so you
    see the typical trajectory of each treatment strategy. Uses one dynamics
    config (default: the first / recommended) so the regimes are comparable.
    """
    if dyn_name is None:
        dyn_name = sorted(raw["dyn"].unique())[0]
    sub = raw[raw["dyn"] == dyn_name].copy()
    r = shaped_reward(sub, w_cell, w_dose, w_smooth)
    sub = sub.assign(_reward=r)
    # per (policy, seed, episode) cumulative series, then aggregate by step
    sub = sub.sort_values(["policy", "seed", "episode", "step"])
    grp = sub.groupby(["policy", "seed", "episode"], group_keys=False)
    sub["cum_reward"] = grp["_reward"].cumsum()
    sub["cum_dose"]   = grp["dose_spent"].cumsum()

    panels = [
        ("cum_reward",   "cumulative reward",        False),
        ("number_tumor", "cancer cells (count)",     False),
        ("dose_spent",   "dose applied / step",      False),
        ("cum_dose",     "cumulative dose (drug burden)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    policies = [p for p in POLICIES if p in sub["policy"].unique()]
    for ax, (col, label, _) in zip(axes, panels):
        for pol in policies:
            d = sub[sub["policy"] == pol]
            # align by step index; mean ± std across seed/episode at each step
            agg = d.groupby("step")[col].agg(["mean", "std"]).reset_index()
            c = _POLICY_COLORS.get(pol, "gray")
            ax.plot(agg["step"], agg["mean"], color=c, label=pol, lw=1.5)
            ax.fill_between(agg["step"],
                            agg["mean"] - agg["std"].fillna(0),
                            agg["mean"] + agg["std"].fillna(0),
                            color=c, alpha=0.15)
        ax.set_xlabel("decision step"); ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        if col == "number_tumor":
            ax.axhline(3, color="k", ls=":", lw=0.8, alpha=0.5)  # termination floor
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Episode evolution by treatment regime  "
                 f"(dyn={dyn_name}, reward w=({w_cell},{w_dose},{w_smooth}))",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "fig_time_series.png"), dpi=130)
    plt.close(fig)


def plot_dynamics_comparison(raw, w_cell, w_dose, w_smooth, outdir):
    """
    The deliverable: one panel PER dynamics config (action_repeat + delta clip),
    each showing mean cumulative return for the three reference policies.

    Lets you read off directly how the RECOMMENDED config (action_repeat=4,
    delta=[1,0.15,0.15,0.05]) compares to alternatives, and whether each config
    preserves the desired ordering (zero_drug ≥ max_drug_fixed; random can win).
    """
    summary = policy_comparison(raw, w_cell, w_dose, w_smooth)
    dyns = list(summary["dyn"].unique())
    pol_order = ["zero_drug", "random", "max_drug_fixed"]
    colors = {"zero_drug": "#264653", "random": "#2a9d8f", "max_drug_fixed": "#e76f51"}

    # ── panel grid: one subplot per dynamics config ──
    n = len(dyns)
    ncol = min(n, 4); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow), squeeze=False)
    for k, d in enumerate(dyns):
        ax = axes[k // ncol][k % ncol]
        sub = summary[summary["dyn"] == d].set_index("policy")
        means = [sub.loc[p, "mean_return"] if p in sub.index else np.nan for p in pol_order]
        errs  = [sub.loc[p, "std_return"]  if p in sub.index else 0.0    for p in pol_order]
        ax.bar(range(len(pol_order)), means, yerr=errs, capsize=4,
               color=[colors[p] for p in pol_order])
        ax.set_xticks(range(len(pol_order)))
        ax.set_xticklabels(pol_order, rotation=20, fontsize=7)
        ax.axhline(0, color="k", lw=0.7)
        gap_zm = (sub.loc["zero_drug", "mean_return"] - sub.loc["max_drug_fixed", "mean_return"]
                  if {"zero_drug", "max_drug_fixed"} <= set(sub.index) else np.nan)
        ax.set_title(f"{d}\nzero−max = {gap_zm:+.2f}", fontsize=8)
        ax.set_ylabel("mean cum. return", fontsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Cumulative return by policy, per dynamics config "
                 f"(w_cell={w_cell}, w_dose={w_dose}, w_smooth={w_smooth})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "fig_dynamics_comparison.png"), dpi=130)
    plt.close(fig)

    # ── single overlay: grouped bars, dyn on x, policy as series ──
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * n), 5))
    width = 0.25
    x = np.arange(n)
    for i, pol in enumerate(pol_order):
        means, errs = [], []
        for d in dyns:
            sub = summary[(summary["dyn"] == d) & (summary["policy"] == pol)]
            means.append(float(sub["mean_return"].iloc[0]) if len(sub) else np.nan)
            errs.append(float(sub["std_return"].iloc[0]) if len(sub) else 0.0)
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3,
               label=pol, color=colors[pol])
    ax.set_xticks(x); ax.set_xticklabels(dyns, rotation=20, fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean cumulative episode return")
    ax.set_title("Dynamics hyperparameter comparison\n"
                 "(want zero_drug ≥ max_drug_fixed; random able to beat max_drug_fixed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_dynamics_overlay.png"), dpi=130)
    plt.close(fig)
    return summary


def composite_score(row):
    """
    Single ranking score combining the four goals. Each term is normalised to
    roughly [0,1] favourable direction; weights reflect the user's priorities:
    smooth policy + dose penalty + T-cell attribution + img>scalars.
    """
    s = 0.0
    # dose penalty: reward when corr is negative (dose suppresses return)
    s += np.tanh(-row["dose_penalty_corr"] * 3)          # want negative -> positive score
    s += np.tanh(row["kill_reward_corr"] * 3)            # want positive
    s += np.tanh(row["attribution_margin"] * 3)          # want positive
    if np.isfinite(row["img_minus_scalar"]):
        s += np.tanh(row["img_minus_scalar"])            # want positive
    s += np.tanh(-row["mean_smooth_cost"])               # want small
    # heavily reward configs where blind dumping does NOT win
    if np.isfinite(row.get("gap_zero_vs_max", np.nan)):
        s += np.tanh(row["gap_zero_vs_max"])             # want >= 0
    return s


# ── correlation helpers ────────────────────────────────────────────────────

def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12 or len(a) < 3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _partial_corr(y, x, z):
    """corr(y, x) controlling for z, via residuals of linear regression on z."""
    y = np.asarray(y, dtype=np.float64); x = np.asarray(x, dtype=np.float64); z = np.asarray(z, dtype=np.float64)
    if len(y) < 4:
        return 0.0
    def resid(t):
        if z.std() < 1e-12:
            return t - t.mean()
        b = np.polyfit(z, t, 1)
        return t - (b[0] * z + b[1])
    return _safe_corr(resid(y), resid(x))


# ──────────────────────────────────────────────────────────────────────────
# 3. Plots
# ──────────────────────────────────────────────────────────────────────────

def make_plots(raw, ranking, outdir):
    # fig 1: dose vs kill colored by attribution gate (mechanism view)
    fig, ax = plt.subplots(figsize=(6, 5))
    attr_gate = raw["n_tcell"] / (raw["n_tcell"] + 8.0)
    sc = ax.scatter(raw["dose_spent"], raw["tumor_killed"], c=attr_gate,
                    cmap="viridis", s=6, alpha=0.5)
    ax.set_xlabel("dose_spent / step"); ax.set_ylabel("tumor cells killed / step")
    ax.set_title("Kill vs dose, coloured by T-cell presence")
    fig.colorbar(sc, ax=ax, label="T-cell attribution gate")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_dose_vs_kill.png"), dpi=130); plt.close(fig)

    # fig 2: smoothness vs w_smooth
    fig, ax = plt.subplots(figsize=(6, 4))
    for (wc, wd), grp in ranking.groupby(["w_cell", "w_dose"]):
        grp = grp.sort_values("w_smooth")
        ax.plot(grp["w_smooth"], grp["mean_smooth_cost"], marker="o", alpha=0.5,
                label=f"wc={wc},wd={wd}")
    ax.set_xlabel("w_smooth"); ax.set_ylabel("mean episode smoothness cost")
    ax.set_title("Smoothness penalty effect")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_smoothness.png"), dpi=130); plt.close(fig)

    # fig 3: img - scalars gap per config (only if both modes were run)
    r = ranking.dropna(subset=["img_minus_scalar"]).sort_values("img_minus_scalar")
    if len(r):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(range(len(r)), r["img_minus_scalar"],
                color=np.where(r["img_minus_scalar"] > 0, "#2a9d8f", "#e76f51"))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("mean img return − mean scalars return")
        ax.set_ylabel("config (sorted)")
        ax.set_title("Does img outperform scalars under each reward?")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_img_vs_scalars.png"), dpi=130); plt.close(fig)

    # fig 4: attribution margin vs dose penalty (the two mechanism goals)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(ranking["dose_penalty_corr"], ranking["attribution_margin"],
                    c=ranking["composite_score"], cmap="plasma", s=40)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("dose-penalty corr (want < 0)")
    ax.set_ylabel("T-cell attribution margin (want > 0)")
    ax.set_title("Mechanism trade-off (color = composite score)")
    fig.colorbar(sc, ax=ax, label="composite score")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_attribution.png"), dpi=130); plt.close(fig)


def write_report(ranking, outdir, raw, tradeoff=None):
    best = ranking.iloc[0]
    lines = [
        "# Reward Function Hyperparameter Analysis\n",
        "\n## Objective: what the trained policy should do\n",
        "1. **smooth** (low action jitter)  2. **decrease cancer cells**  "
        "3. **minimum drug introduced**.",
        "The reward weights trade these off:",
        "`w_cell`↑ → more killing · `w_dose`↑ → less drug · `w_smooth`↑ → smoother.",
        "See `fig_outcome_tradeoff.png` / `outcome_tradeoff.csv` for the three",
        "outcomes in real units (tumor_reduction, total_dose, smooth_cost) per config.\n",
        f"- raw steps logged: **{len(raw)}**  across "
        f"**{raw['obs_mode'].nunique()}** observation modes, "
        f"**{raw['seed'].nunique()}** seeds, "
        f"**{raw.groupby(['obs_mode','seed'])['episode'].nunique().sum()}** episodes\n",
        "\n## Goals & how they are measured\n",
        "| Goal | Metric | Target |",
        "|------|--------|--------|",
        "| Penalise dosing | `dose_penalty_corr` (partial corr of return vs dose, controlling kill) | < 0 |",
        "| Encourage killing | `kill_reward_corr` (corr return vs tumor killed) | > 0 |",
        "| Credit T-cell mechanism | `attribution_margin` (corr(r, T-cell kill) − corr(r, dose)) | > 0 |",
        "| img beats scalars | `img_minus_scalar` (mean img return − mean scalar return) | > 0 |",
        "| Smooth policy | `mean_smooth_cost` | small |",
        "\n## Headline: cumulative return by fixed policy\n",
        "See `policy_comparison.csv` and `fig_policy_comparison.png`.",
        "A correct reward must satisfy:",
        "- `gap_zero_vs_max = return(zero_drug) − return(max_drug_fixed) ≥ 0`  "
        "(blind dumping must not pay, because the drug doesn't kill — T cells do)",
        "- `gap_random_vs_max > 0`  (a policy that explores/targets can beat always-max-dose)\n",
        f"Best config gaps: zero_vs_max = {best['gap_zero_vs_max']:+.3f}, "
        f"random_vs_max = {best['gap_random_vs_max']:+.3f}\n",
        "\n## Recommended configuration\n",
        "Reward form (unchanged from `wrapper_tip.py`, dose weight exposed):\n",
        "`reward = w_cell * r_cancer_cells - w_dose * dose_spent - w_smooth * smooth_penalty`\n",
        f"```\nw_cell   = {best['w_cell']}\n"
        f"w_dose   = {best['w_dose']}   # = 1.0 reproduces the current env exactly\n"
        f"w_smooth = {best['w_smooth']}\n```\n",
        "Metrics for this config:\n",
        f"- dose_penalty_corr  = {best['dose_penalty_corr']:+.3f}  (want < 0)",
        f"- kill_reward_corr   = {best['kill_reward_corr']:+.3f}  (want > 0)",
        f"- attribution_margin = {best['attribution_margin']:+.3f}  (diagnostic: reward tracks T-cell kills more than dose)",
        (f"- img_minus_scalar   = {best['img_minus_scalar']:+.3f}  (want > 0)"
         if np.isfinite(best['img_minus_scalar'])
         else "- img_minus_scalar   = n/a  (single observation mode run)"),
        f"- mean_smooth_cost   = {best['mean_smooth_cost']:.4f}  (smaller better)",
        f"- composite_score    = {best['composite_score']:+.3f}",
        "\n## How to apply in the wrapper\n",
        "Only the weights change; the reward expression in `wrapper_tip.py` stays:\n",
        "```python",
        f"reward = {best['w_cell']} * r_cancer_cells "
        f"- {best['w_dose']} * dose_spent "
        f"- {best['w_smooth']} * smooth_penalty",
        "```",
        "T-cell / macrophage counts are logged and plotted for attribution analysis,",
        "but are NOT part of the reward.\n",
        "\n## Top 15 configurations\n",
        ranking.head(15).to_markdown(index=False, floatfmt="+.3f"),
    ]
    if tradeoff is not None and not tradeoff.empty:
        # rank by the three goals jointly: high reduction, low dose, low jitter
        t = tradeoff.drop_duplicates(subset=["dyn"]).copy()
        def _n(col, invert=False):
            v = t[col]; rng = (v.max() - v.min()) or 1.0
            z = (v - v.min()) / rng
            return (1 - z) if invert else z
        t["goal_score"] = _n("tumor_reduction") + _n("total_dose", True) + _n("smooth_cost", True)
        t = t.sort_values("goal_score", ascending=False)
        lines += [
            "\n## Three-goal outcome trade-off (real units, on 'random' probe)\n",
            "One row per dynamics config. Ranked by: high tumor_reduction + "
            "low total_dose + low smooth_cost.\n",
            t[["dyn", "tumor_reduction", "total_dose", "smooth_cost", "goal_score"]]
                .to_markdown(index=False, floatfmt=".3f"),
        ]
    with open(os.path.join(outdir, "reward_sweep_report.md"), "w") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────
# 4. CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Quantitative reward-function hyperparameter analysis.")
    p.add_argument("--obs_modes", nargs="+",
                   default=["img_mc_cells_substrates", "scalars_macrophages"],
                   help="observation modes to compare (include at least one 'img' and one scalar)")
    p.add_argument("--seeds", nargs="+", type=int, default=[200, 201, 202])
    p.add_argument("--episodes_per_seed", type=int, default=4)
    p.add_argument("--max_time", type=float, default=7200.0,
                   help="episode max_time; MUST match run.py --max_time_episode (7200)")
    p.add_argument("--settingxml", default="config/PhysiCell_settings.xml")
    p.add_argument("--settingcells", default="config/cells.csv")
    p.add_argument("--outdir", default="reward_analysis_results")
    p.add_argument("--raw_csv", default=None,
                   help="reuse a previously saved raw_components.csv instead of re-running sims")
    p.add_argument("--replot", action="store_true",
                   help="skip rollouts, load --raw_csv, just sweep + plot")
    p.add_argument("--parallel", action="store_true",
                   help="run seeds in parallel via SubprocVecEnv (one worker per seed)")
    # hyperparameter grid
    p.add_argument("--grid_w_cell",  nargs="+", type=float, default=[0.3, 0.5, 1.0])
    p.add_argument("--grid_w_dose",  nargs="+", type=float, default=[0.5, 1.0, 2.0])
    p.add_argument("--grid_w_smooth",nargs="+", type=float, default=[0.0, 0.02, 0.1])
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── obtain raw component trajectories ──
    if args.raw_csv and (args.replot or os.path.exists(args.raw_csv)):
        print(f"[load] raw components from {args.raw_csv}")
        raw = pd.read_csv(args.raw_csv)
    else:
        t0 = time.time()
        rollout_fn = run_rollouts_parallel if args.parallel else run_rollouts
        raw = rollout_fn(args.obs_modes, args.seeds, args.episodes_per_seed,
                         args.max_time, args.settingxml, args.settingcells)
        raw_path = os.path.join(args.outdir, "raw_components.csv")
        raw.to_csv(raw_path, index=False)
        print(f"[rollout] done in {time.time()-t0:.1f}s -> {raw_path} ({len(raw)} steps)")

    if raw.empty:
        sys.exit("No raw data collected; aborting.")

    # ── sweep ──
    grid = list(itertools.product(
        args.grid_w_cell, args.grid_w_dose, args.grid_w_smooth))
    print(f"[sweep] {len(grid)} configurations over {len(raw)} steps")
    results = [evaluate_config(raw, *g) for g in grid]
    ranking = pd.DataFrame(results)
    ranking["composite_score"] = ranking.apply(composite_score, axis=1)
    ranking = ranking.sort_values("composite_score", ascending=False).reset_index(drop=True)

    ranking.to_csv(os.path.join(args.outdir, "reward_sweep_ranking.csv"), index=False)
    make_plots(raw, ranking, args.outdir)

    # ── outcome trade-off: the three goals in real units ──
    # smooth policy + cancer decrease + minimum drug, per reward config.
    to = plot_outcome_tradeoff(raw, grid, args.outdir)
    if not to.empty:
        print("[outcome] tumor_reduction / total_dose / smooth_cost per config -> "
              "outcome_tradeoff.csv")

    # ── headline policy comparison: random vs 0-drug vs max-drug-fixed ──
    # use the best config + the literature defaults so the bar chart is readable
    best = ranking.iloc[0]
    compare_cfgs = [
        (best["w_cell"], best["w_dose"], best["w_smooth"]),
        (0.3, 1.0, 0.02),   # current wrapper default (w_cell=0.3, dose coeff=1)
    ]
    # de-duplicate while preserving order
    seen = set(); compare_cfgs = [c for c in compare_cfgs if not (c in seen or seen.add(c))]
    plot_policy_comparison(raw, compare_cfgs, args.outdir)

    # ── dynamics comparison: one plot per (action_repeat, delta) config ──
    # rendered for both the best reward and the env-default reward weights.
    n_dyn = raw["dyn"].nunique()
    print(f"[dynamics] {n_dyn} dynamics config(s): {sorted(raw['dyn'].unique())}")
    plot_dynamics_comparison(raw, *compare_cfgs[0], args.outdir)
    # episode evolution: cumulative reward / cancer cells / dose regime per policy
    plot_time_series(raw, *compare_cfgs[0], args.outdir)

    pc_rows = []
    for cfg in compare_cfgs:
        s = policy_comparison(raw, *cfg)
        s = s.assign(w_cell=cfg[0], w_dose=cfg[1], w_smooth=cfg[2],
                     gap_zero_vs_max=s.attrs["gap_zero_vs_max"],
                     gap_random_vs_max=s.attrs["gap_random_vs_max"])
        pc_rows.append(s)
    # columns now: dyn, policy, mean_return, std_return, n, w_cell, w_dose, w_smooth, gaps
    pd.concat(pc_rows).to_csv(
        os.path.join(args.outdir, "policy_comparison.csv"), index=False)

    write_report(ranking, args.outdir, raw, tradeoff=to)

    print(f"\n[done] results in {args.outdir}/")
    print(ranking.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
