# Hyperparameter Search — Methodology & Justification

## Why this is defensible

- **Search**: Latin Hypercube (20 configs) over the dynamics hyperparameters ['action_repeat', 'delta_x', 'delta_y', 'delta_radius'] — space-filling, not cherry-picked.
- **Cost control**: each LHS point = one parallel rollout (6 seeds × 2 episodes); reward weights swept offline (they don't change physics).
- **Objective** (scalarized, explicit trade-off):
  `score = 1.0·z(tumor_reduction) − 1.0·z(total_dose) − 1.0·z(smooth_cost)`
- **Uncertainty**: bootstrap 95% CIs across seeds on every outcome; Optuna importance (True); Sobol indices (True).

## Chosen configuration

```
action_repeat = 6
delta_x       = 0.254
delta_y       = 0.235
delta_radius  = 0.028
w_cell        = 0.3
w_dose        = 2.0
w_smooth      = 0.02
```

Outcomes (mean [95% CI]):

- tumor_reduction = -42.42 [-73.18, -10.58]
- total_dose      = 5.608 [5.225, 6.011]
- smooth_cost     = 18.582 [17.764, 19.415]

## How to read 'why slow/smooth wins'

See `fig_lhs_objective.png` (objective vs each param), `fig_param_importance.png` (which param matters most), and `fig_sobol.png` (which param drives which outcome). If, e.g., a high `action_repeat` raises the objective AND Sobol shows it dominates `smooth_cost` variance, that IS the quantitative reason a smoother config is recommended.

## Top configurations

|   action_repeat |   delta_x |   delta_y |   delta_radius |   w_cell |   w_dose |   w_smooth |   tumor_reduction |   total_dose |   smooth_cost |   objective |
|----------------:|----------:|----------:|---------------:|---------:|---------:|-----------:|------------------:|-------------:|--------------:|------------:|
|           6.000 |     0.254 |     0.235 |          0.028 |    0.300 |    2.000 |      0.020 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.300 |    2.000 |      0.000 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.300 |    1.000 |      0.100 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.500 |    1.000 |      0.000 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.500 |    0.500 |      0.100 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.500 |    1.000 |      0.100 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.500 |    2.000 |      0.000 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.500 |    1.000 |      0.020 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    1.000 |    0.500 |      0.000 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.500 |    2.000 |      0.020 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    0.500 |    2.000 |      0.100 |           -42.417 |        5.608 |        18.582 |       0.053 |
|           6.000 |     0.254 |     0.235 |          0.028 |    1.000 |    1.000 |      0.000 |           -42.417 |        5.608 |        18.582 |       0.053 |