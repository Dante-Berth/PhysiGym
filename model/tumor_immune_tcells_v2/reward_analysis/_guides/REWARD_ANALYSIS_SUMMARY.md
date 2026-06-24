# Reward Analysis Summary Report

## Key Findings

### 1. Action Repeat Dominates Everything
- **Importance for objective**: 0.82 (out of 1.0)
- **Why**: Controls timescale on which agent acts; must match immune-response window (~5-7 min)
- **Optimal value**: 6 (holds dose steady for ~6 minutes)

### 2. Spatial Parameters (delta_x/y/radius) are Secondary
- **Combined importance**: ~0.15
- **Safe to round**: 0.254 → 0.25, 0.235 → 0.25, 0.028 → 0.03
- **Least important**: delta_x and delta_radius (< 0.05 each)

### 3. Reward Weights are FLEXIBLE
- Once dynamics are fixed, reward weights have little impact on outcomes
- This is because they don't change physics—they only reweight the same trajectories
- **Recommended defaults**: w_cell=0.3, w_dose=2.0, w_smooth=0.02
- But you can adjust these during RL training without re-running expensive dynamics search

### 4. Top Configuration
```
Dynamics:
  action_repeat = 6
  delta_x = 0.25 (0.254)
  delta_y = 0.25 (0.235)
  delta_radius = 0.03 (0.028)

Reward weights:
  w_cell = 0.3
  w_dose = 2.0
  w_smooth = 0.02

Outcomes:
  Tumor reduction: -42.4 cells [95% CI: -73.2 to -10.6] ✅ statistically significant
  Total dose: 5.61 units (minimal drug waste)
  Smoothness cost: 18.58 (low action jitter)
```

## Figures Generated

### Dynamics Analysis
- `fig_action_repeat_sensitivity.png` — Shows why AR=6 peaks (tumor/smoothness/objective)
- `fig_lhs_objective.png` — All parameters vs objective response
- `fig_param_importance.png` — Optuna importance ranking (dynamics)
- `fig_sobol.png` — Sobol sensitivity (variance decomposition by parameter)
- `fig_outcome_cis.png` — Bootstrap 95% CIs across seeds (top configs)

### Reward Weight Analysis
- `fig_reward_weight_correlation.png` — Heatmap of reward weights vs objective
- `fig_reward_weight_distributions.png` — Where top configs cluster in weight space
- `fig_optuna_param_importance_rewards_static.png` — Importance ranking (rewards)
- `fig_dynamics_vs_rewards_importance.png` — Side-by-side: dynamics >> rewards

### Time Series
- `fig_time_series.png` — Episode trajectories (cumulative reward, cancer count, dose)

## How to Use These Results

1. **Training**: Use action_repeat=6, delta_x/y/radius=0.25/0.25/0.03
2. **Reward weights**: Start with w_cell=0.3, w_dose=2.0, w_smooth=0.02; adjust if needed
3. **Paper justification**: See HYPERPARAMETER_DEFENSE.md for detailed reasoning
4. **Reproducibility**: All hyperparameters and results are in search_results.csv

## References

- **Search method**: Latin Hypercube Design (20 configs) × 6 seeds × 2 episodes
- **Search space**:
  - Dynamics: action_repeat ∈ [1,8], delta_x/y/radius ∈ [0.05,0.30]
  - Rewards: w_cell ∈ [0.1,1.0], w_dose ∈ [0.5,2.0], w_smooth ∈ [0.0,0.1]
- **Objective**: Maximize tumor_reduction, minimize total_dose, minimize smooth_cost
- **Sensitivity**: Optuna importance + Sobol indices (RandomForest surrogate)
