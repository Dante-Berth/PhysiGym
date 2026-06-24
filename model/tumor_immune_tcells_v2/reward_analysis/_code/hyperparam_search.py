#####
# title: hyperparam_search.py
#
# Defensible hyperparameter research for the PhysiCell drug-dosing reward.
#
# Answers the question a reviewer will ask: "HOW did you choose these
# hyperparameters, and WHY?" — with a space-filling search, a scalarized
# multi-goal objective, and uncertainty quantification.
#
# Method
# ──────
# 1. LATIN HYPERCUBE SAMPLING (scipy.stats.qmc) over the DYNAMICS
#    hyperparameters that actually change the trajectory physics:
#       action_repeat ∈ [1, 8]   (int)
#       delta_x       ∈ [0.05, 0.30]
#       delta_y       ∈ [0.05, 0.30]
#       delta_radius  ∈ [0.02, 0.10]
#    LHS gives even coverage of this 4-D space with far fewer points than a
#    full grid — each point is ONE parallel rollout (seed-replicated).
#
# 2. OFFLINE REWARD-WEIGHT SWEEP per sampled config (free — weights don't
#    change physics, only re-score logged trajectories).
#
# 3. SCALARIZED 3-GOAL OBJECTIVE (the user's target policy):
#       score =  α · z(tumor_reduction)        # decrease cancer  (↑)
#              − β · z(total_dose)              # minimum drug      (↓)
#              − γ · z(smooth_cost)             # smooth policy     (↓)
#    where z(·) is min-max normalisation across all evaluated configs so the
#    three incommensurable outcomes are comparable. α,β,γ are explicit and
#    reported, so the trade-off you committed to is transparent.
#
# 4. UNCERTAINTY QUANTIFICATION:
#    - bootstrap 95% CIs across seeds for each outcome (always)
#    - Optuna study: records every trial, ranks params by importance
#    - SALib Sobol indices: which hyperparameter drives which outcome
#      (first-order + total-order), if SALib is available.
#
# Outputs (--outdir):
#   lhs_design.csv            the sampled dynamics configs
#   search_results.csv        per-config outcomes + CIs + objective score
#   fig_lhs_objective.png     objective vs each hyperparameter (response curves)
#   fig_outcome_cis.png       3 outcomes with seed CIs, ranked
#   fig_param_importance.png  Optuna param importances for the objective
#   fig_sobol.png             Sobol first/total indices per outcome (if SALib)
#   search_report.md          the "why these hyperparameters" writeup
#####

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import qmc

# reuse the validated rollout + offline reward machinery
from reward_analysis import (
    run_rollouts_parallel, shaped_reward, episode_metrics, POLICIES,
    plot_time_series,
)

# optional deps
try:
    import optuna
    _HAVE_OPTUNA = True
except Exception:
    _HAVE_OPTUNA = False
try:
    from SALib.sample import latin as salib_latin
    from SALib.analyze import sobol as salib_sobol
    from SALib.sample import saltelli
    _HAVE_SALIB = True
except Exception:
    _HAVE_SALIB = False


# ──────────────────────────────────────────────────────────────────────────
# search space
# ──────────────────────────────────────────────────────────────────────────
DYN_BOUNDS = {
    "action_repeat": (1, 8),       # int
    "delta_x":       (0.05, 0.30),
    "delta_y":       (0.05, 0.30),
    "delta_radius":  (0.02, 0.10),
}
DYN_NAMES = list(DYN_BOUNDS.keys())


def lhs_design(n_samples, seed=0):
    """Latin Hypercube sample of dynamics configs. Returns list of dyn dicts."""
    sampler = qmc.LatinHypercube(d=len(DYN_NAMES), seed=seed)
    unit = sampler.random(n=n_samples)               # (n, 4) in [0,1]
    lo = np.array([DYN_BOUNDS[k][0] for k in DYN_NAMES])
    hi = np.array([DYN_BOUNDS[k][1] for k in DYN_NAMES])
    scaled = qmc.scale(unit, lo, hi)
    designs = []
    for i, row in enumerate(scaled):
        ar = int(round(row[0]))
        designs.append(dict(
            name=f"lhs_{i:03d}",
            action_repeat=ar,
            delta=[1.0, float(row[1]), float(row[2]), float(row[3])],
            # keep raw params for analysis
            _params={"action_repeat": ar, "delta_x": float(row[1]),
                     "delta_y": float(row[2]), "delta_radius": float(row[3])},
        ))
    return designs


# ──────────────────────────────────────────────────────────────────────────
# objective + UQ
# ──────────────────────────────────────────────────────────────────────────
def _z(series):
    s = np.asarray(series, dtype=np.float64)
    rng = (s.max() - s.min()) or 1.0
    return (s - s.min()) / rng


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """95% bootstrap CI for the mean of `values` (across seeds/episodes)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan,) * 3
    rng = rng or np.random.default_rng(0)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def config_outcomes(raw, dyn_name, w_cell, w_dose, w_smooth, policy="random"):
    """
    Mean + bootstrap CI of the 3 outcomes for one dynamics config under one
    reward weighting, computed across seeds/episodes of the probe policy.
    """
    r = shaped_reward(raw, w_cell, w_dose, w_smooth)
    ep = episode_metrics(raw, r)
    ep = ep[(ep["dyn"] == dyn_name) & (ep["policy"] == policy)]
    out = {}
    for col in ["tumor_reduction", "total_dose", "smooth_cost"]:
        m, lo, hi = bootstrap_ci(ep[col].to_numpy())
        out[col] = m
        out[col + "_lo"] = lo
        out[col + "_hi"] = hi
    out["n_episodes"] = len(ep)
    return out


def scalarized_score(df, alpha, beta, gamma):
    """score = α·z(reduction) − β·z(dose) − γ·z(smooth). Higher = better."""
    return (alpha * _z(df["tumor_reduction"])
            - beta  * _z(df["total_dose"])
            - gamma * _z(df["smooth_cost"]))


# ──────────────────────────────────────────────────────────────────────────
# main search
# ──────────────────────────────────────────────────────────────────────────
def run_search(args):
    os.makedirs(args.outdir, exist_ok=True)

    # 1. LHS design over dynamics
    designs = lhs_design(args.n_lhs, seed=args.lhs_seed)
    lhs_df = pd.DataFrame([d["_params"] | {"name": d["name"]} for d in designs])
    lhs_df.to_csv(os.path.join(args.outdir, "lhs_design.csv"), index=False)
    print(f"[lhs] {len(designs)} dynamics configs sampled:")
    print(lhs_df.to_string(index=False))

    # 2. one parallel rollout per design (seed-replicated for UQ).
    #    CHECKPOINTED: each design is rolled out separately and its raw rows
    #    appended to raw_components.csv immediately, so a stall on config K
    #    never loses configs 0..K-1, and a re-launch resumes where it stopped.
    raw_path = os.path.join(args.outdir, "raw_components.csv")
    roll_policies = args.policies if args.policies else [args.policy]

    if args.raw_csv and os.path.exists(args.raw_csv):
        print(f"[load] reusing raw rollouts from {args.raw_csv}")
        raw = pd.read_csv(args.raw_csv)
    else:
        # resume: which designs are already on disk?
        done = set()
        if os.path.exists(raw_path):
            try:
                done = set(pd.read_csv(raw_path, usecols=["dyn"])["dyn"].unique())
                print(f"[resume] {len(done)} configs already in {raw_path}: {sorted(done)}")
            except Exception as e:
                print(f"[resume] could not read existing checkpoint ({e}); starting fresh")
                done = set()

        for i, d in enumerate(designs):
            if d["name"] in done:
                print(f"[skip {i+1}/{len(designs)}] {d['name']} already done")
                continue
            t0 = time.time()
            print(f"[rollout {i+1}/{len(designs)}] {d['name']} "
                  f"repeat={d['action_repeat']} delta={d['delta']}", flush=True)
            part = run_rollouts_parallel(
                obs_modes=[args.obs_mode],
                seeds=list(args.seeds),
                episodes_per_seed=args.episodes_per_seed,
                max_time=args.max_time,
                base_xml=args.settingxml,
                base_cells=args.settingcells,
                policies=roll_policies,
                dyn=[d],
            )
            if part is None or part.empty:
                print(f"[warn] {d['name']} produced no rows; continuing", flush=True)
                continue
            # append checkpoint (write header only the first time)
            header = not os.path.exists(raw_path)
            part.to_csv(raw_path, mode="a", header=header, index=False)
            print(f"[ckpt {i+1}/{len(designs)}] {d['name']}: {len(part)} steps "
                  f"in {time.time()-t0:.1f}s -> appended to {raw_path}", flush=True)

        if not os.path.exists(raw_path):
            raise SystemExit("no rollout data collected")
        raw = pd.read_csv(raw_path)

    if raw.empty:
        raise SystemExit("no rollout data collected")

    # 3. offline reward-weight sweep × outcomes per config
    rows = []
    for d in designs:
        name = d["name"]
        if name not in set(raw["dyn"].unique()):
            continue  # config may have died in rollout
        for wc in args.grid_w_cell:
            for wd in args.grid_w_dose:
                for ws in args.grid_w_smooth:
                    oc = config_outcomes(raw, name, wc, wd, ws, policy=args.policy)
                    rows.append(d["_params"] | {
                        "name": name, "w_cell": wc, "w_dose": wd, "w_smooth": ws,
                        **oc,
                    })
    res = pd.DataFrame(rows)
    res["objective"] = scalarized_score(res, args.alpha, args.beta, args.gamma)
    res = res.sort_values("objective", ascending=False).reset_index(drop=True)
    res.to_csv(os.path.join(args.outdir, "search_results.csv"), index=False)

    best = res.iloc[0]
    print(f"\n[best] {best['name']}  "
          f"action_repeat={best['action_repeat']}  "
          f"delta=[{best['delta_x']:.3f},{best['delta_y']:.3f},{best['delta_radius']:.3f}]  "
          f"w=({best['w_cell']},{best['w_dose']},{best['w_smooth']})  "
          f"objective={best['objective']:.3f}")

    # 4. plots + UQ
    _plot_response_curves(res, args.outdir)
    _plot_outcome_cis(res, args.outdir)
    # episode evolution under the BEST dynamics config: cumulative reward,
    # cancer-cell count, and the dose regime (incl. cosine pulse) per policy.
    plot_time_series(raw, best["w_cell"], best["w_dose"], best["w_smooth"],
                     args.outdir, dyn_name=best["name"])
    if _HAVE_OPTUNA:
        _optuna_importance(res, args, args.outdir)
    if _HAVE_SALIB and args.sobol:
        _sobol_analysis(res, args.outdir)
    _write_report(res, args, args.outdir)
    print(f"\n[done] {args.outdir}/  (optuna={_HAVE_OPTUNA}, salib={_HAVE_SALIB})")
    return res


def _plot_response_curves(res, outdir):
    """Objective vs each hyperparameter — the 'why' at a glance."""
    params = DYN_NAMES + ["w_cell", "w_dose", "w_smooth"]
    fig, axes = plt.subplots(1, len(params), figsize=(3 * len(params), 3.2), squeeze=False)
    for j, p in enumerate(params):
        ax = axes[0][j]
        ax.scatter(res[p], res["objective"], s=14, alpha=0.5, color="#2a9d8f")
        # binned mean trend
        try:
            q = pd.qcut(res[p], min(6, res[p].nunique()), duplicates="drop")
            g = res.groupby(q, observed=True).agg(x=(p, "mean"), y=("objective", "mean"))
            ax.plot(g["x"], g["y"], "-o", color="#e76f51", ms=4)
        except Exception:
            pass
        ax.set_xlabel(p, fontsize=8); ax.set_title(p, fontsize=8)
    axes[0][0].set_ylabel("objective (higher better)")
    fig.suptitle("Objective response to each hyperparameter", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "fig_lhs_objective.png"), dpi=130)
    plt.close(fig)


def _plot_outcome_cis(res, outdir):
    """Top configs: the 3 outcomes with bootstrap seed CIs."""
    top = res.drop_duplicates("name").head(12)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [("tumor_reduction", "tumor reduction (↑)", "#2a9d8f"),
             ("total_dose",      "total dose (↓)",      "#e76f51"),
             ("smooth_cost",     "smoothness cost (↓)", "#264653")]
    y = np.arange(len(top))
    for ax, (col, label, c) in zip(axes, specs):
        err = np.array([top[col] - top[col + "_lo"], top[col + "_hi"] - top[col]])
        ax.barh(y, top[col], xerr=np.abs(err), color=c, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels(top["name"], fontsize=6)
        ax.set_xlabel(label, fontsize=9); ax.invert_yaxis()
    fig.suptitle("Outcomes with bootstrap 95% CI across seeds (top configs)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, "fig_outcome_cis.png"), dpi=130)
    plt.close(fig)


def _optuna_importance(res, args, outdir):
    """
    Replay the LHS evaluations into an Optuna study so we get principled
    parameter-importance for the scalarized objective (fANOVA/MDI).
    """
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    params = DYN_NAMES + ["w_cell", "w_dose", "w_smooth"]
    for _, row in res.iterrows():
        dist = {
            "action_repeat": optuna.distributions.IntDistribution(1, 8),
            "delta_x": optuna.distributions.FloatDistribution(0.05, 0.30),
            "delta_y": optuna.distributions.FloatDistribution(0.05, 0.30),
            "delta_radius": optuna.distributions.FloatDistribution(0.02, 0.10),
            "w_cell": optuna.distributions.FloatDistribution(
                min(args.grid_w_cell), max(args.grid_w_cell) + 1e-9),
            "w_dose": optuna.distributions.FloatDistribution(
                min(args.grid_w_dose), max(args.grid_w_dose) + 1e-9),
            "w_smooth": optuna.distributions.FloatDistribution(
                min(args.grid_w_smooth), max(args.grid_w_smooth) + 1e-9),
        }
        trial = optuna.trial.create_trial(
            params={p: row[p] for p in params},
            distributions=dist,
            value=float(row["objective"]),
        )
        study.add_trial(trial)

    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"[optuna] importance failed: {e}")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(imp.keys())[::-1]; vals = list(imp.values())[::-1]
    ax.barh(names, vals, color="#2a9d8f")
    ax.set_xlabel("importance for objective")
    ax.set_title("Optuna parameter importance (higher = more decisive)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_param_importance.png"), dpi=130)
    plt.close(fig)
    pd.Series(imp).to_csv(os.path.join(outdir, "param_importance.csv"))
    print("[optuna] importance:", {k: round(v, 3) for k, v in imp.items()})


def _sobol_analysis(res, outdir):
    """
    Sobol variance decomposition: how much of each OUTCOME's variance is
    explained by each dynamics hyperparameter (first-order S1 + total ST).
    Fits a quick surrogate (RandomForest) on the LHS points, then samples it
    with a Saltelli design — standard practice when the true sim is expensive.
    """
    from sklearn.ensemble import RandomForestRegressor

    problem = {
        "num_vars": len(DYN_NAMES),
        "names": DYN_NAMES,
        "bounds": [list(DYN_BOUNDS[k]) for k in DYN_NAMES],
    }
    # one surrogate per outcome on the (dedup by config) LHS data
    base = res.drop_duplicates("name")
    X = base[DYN_NAMES].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, outcome in zip(axes, ["tumor_reduction", "total_dose", "smooth_cost"]):
        y = base[outcome].to_numpy()
        if len(np.unique(y)) < 3:
            ax.set_title(f"{outcome}\n(insufficient variance)"); continue
        rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
        param_values = saltelli.sample(problem, 1024)
        Y = rf.predict(param_values)
        Si = salib_sobol.analyze(problem, Y, print_to_console=False)
        idx = np.arange(len(DYN_NAMES)); w = 0.38
        ax.bar(idx - w/2, Si["S1"], w, label="S1 (first-order)", color="#2a9d8f")
        ax.bar(idx + w/2, Si["ST"], w, label="ST (total)", color="#e76f51")
        ax.set_xticks(idx); ax.set_xticklabels(DYN_NAMES, rotation=25, fontsize=7)
        ax.set_title(outcome, fontsize=9); ax.legend(fontsize=6)
    fig.suptitle("Sobol sensitivity: variance of each outcome explained by each param",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "fig_sobol.png"), dpi=130)
    plt.close(fig)
    print("[sobol] done")


def _write_report(res, args, outdir):
    best = res.iloc[0]
    base = res.drop_duplicates("name")
    lines = [
        "# Hyperparameter Search — Methodology & Justification\n",
        "## Why this is defensible\n",
        f"- **Search**: Latin Hypercube ({args.n_lhs} configs) over the dynamics "
        f"hyperparameters {DYN_NAMES} — space-filling, not cherry-picked.",
        f"- **Cost control**: each LHS point = one parallel rollout "
        f"({len(args.seeds)} seeds × {args.episodes_per_seed} episodes); reward "
        "weights swept offline (they don't change physics).",
        f"- **Objective** (scalarized, explicit trade-off):",
        f"  `score = {args.alpha}·z(tumor_reduction) − {args.beta}·z(total_dose) "
        f"− {args.gamma}·z(smooth_cost)`",
        f"- **Uncertainty**: bootstrap 95% CIs across seeds on every outcome; "
        f"Optuna importance ({_HAVE_OPTUNA}); Sobol indices ({_HAVE_SALIB and args.sobol}).\n",
        "## Chosen configuration\n",
        f"```\naction_repeat = {int(best['action_repeat'])}\n"
        f"delta_x       = {best['delta_x']:.3f}\n"
        f"delta_y       = {best['delta_y']:.3f}\n"
        f"delta_radius  = {best['delta_radius']:.3f}\n"
        f"w_cell        = {best['w_cell']}\n"
        f"w_dose        = {best['w_dose']}\n"
        f"w_smooth      = {best['w_smooth']}\n```\n",
        "Outcomes (mean [95% CI]):\n",
        f"- tumor_reduction = {best['tumor_reduction']:.2f} "
        f"[{best['tumor_reduction_lo']:.2f}, {best['tumor_reduction_hi']:.2f}]",
        f"- total_dose      = {best['total_dose']:.3f} "
        f"[{best['total_dose_lo']:.3f}, {best['total_dose_hi']:.3f}]",
        f"- smooth_cost     = {best['smooth_cost']:.3f} "
        f"[{best['smooth_cost_lo']:.3f}, {best['smooth_cost_hi']:.3f}]\n",
        "## How to read 'why slow/smooth wins'\n",
        "See `fig_lhs_objective.png` (objective vs each param), "
        "`fig_param_importance.png` (which param matters most), and "
        "`fig_sobol.png` (which param drives which outcome). If, e.g., a high "
        "`action_repeat` raises the objective AND Sobol shows it dominates "
        "`smooth_cost` variance, that IS the quantitative reason a smoother "
        "config is recommended.\n",
        "## Top configurations\n",
        res.head(12)[DYN_NAMES + ["w_cell", "w_dose", "w_smooth",
                                  "tumor_reduction", "total_dose", "smooth_cost",
                                  "objective"]].to_markdown(index=False, floatfmt=".3f"),
    ]
    with open(os.path.join(outdir, "search_report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="LHS + Optuna + Sobol reward hyperparameter search.")
    p.add_argument("--n_lhs", type=int, default=12, help="number of LHS dynamics configs")
    p.add_argument("--lhs_seed", type=int, default=0)
    p.add_argument("--obs_mode", default="scalars_macrophages")
    p.add_argument("--policy", default="random", choices=POLICIES,
                   help="policy the OBJECTIVE is evaluated on")
    p.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES,
                   help="all probe policies to roll out (for time-series regimes)")
    p.add_argument("--seeds", nargs="+", type=int, default=[200, 201, 202, 203])
    p.add_argument("--episodes_per_seed", type=int, default=2)
    p.add_argument("--max_time", type=float, default=7200.0,
                   help="MUST match run.py --max_time_episode (7200)")
    p.add_argument("--settingxml", default="config/PhysiCell_settings.xml")
    p.add_argument("--settingcells", default="config/cells.csv")
    p.add_argument("--outdir", default="hyperparam_search_results")
    p.add_argument("--raw_csv", default=None, help="reuse saved raw_components.csv")
    # objective weights (explicit trade-off)
    p.add_argument("--alpha", type=float, default=1.0, help="weight on tumor reduction")
    p.add_argument("--beta",  type=float, default=1.0, help="weight on dose penalty")
    p.add_argument("--gamma", type=float, default=1.0, help="weight on smoothness")
    # offline reward-weight grid
    p.add_argument("--grid_w_cell",  nargs="+", type=float, default=[0.3, 0.5, 1.0])
    p.add_argument("--grid_w_dose",  nargs="+", type=float, default=[0.5, 1.0, 2.0])
    p.add_argument("--grid_w_smooth", nargs="+", type=float, default=[0.0, 0.02, 0.1])
    p.add_argument("--sobol", action="store_true", help="run SALib Sobol analysis")
    args = p.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
