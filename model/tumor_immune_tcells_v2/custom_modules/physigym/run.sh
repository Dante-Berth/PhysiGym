#!/usr/bin/env bash
# Launch one SAC run per observation mode, sequentially.
# Uses best hyperparameters from reward_analysis: w_cell=0.3, w_dose=2.0, w_smooth=0.0
# Source: reward_analysis composite_score=2.972 (best overall)

SEEDS=(123)

OBS_MODES=(
  img_mc_cells_substrates
  img_mc_cells
  scalars_macrophages
  img_mc_cells_m1m2
  img_mc_cells_substrates_m1m2
)

# Best hyperparameters from reward_analysis
W_CELL=0.3
W_DOSE=2.0
W_SMOOTH=0.0

# ── 2. SAC training runs (one per obs mode) ─────────────────────
for seed in "${SEEDS[@]}"; do
  for obs in "${OBS_MODES[@]}"; do
    echo "============================================================"
    echo "  SAC  seed=${seed}  observation_mode=${obs}"
    echo "  Hyperparameters: w_cell=${W_CELL} w_dose=${W_DOSE} w_smooth=${W_SMOOTH}"
    echo "============================================================"
    python custom_modules/physigym/physigym/envs/run.py \
      --seed           "${seed}"  \
      --observation_mode "${obs}" \
      --action_mode    targeted   \
      --w_cell         ${W_CELL}  \
      --w_dose         ${W_DOSE}  \
      --w_smooth       ${W_SMOOTH} \
      --action_repeat  6          \
      --delta_x        0.25       \
      --delta_y        0.25       \
      --delta_radius   0.03       \
      --total_timesteps 100000    \
      --wandb          true       \
      --name           "best_hyperparameters_SAC_${obs}_w_cell=${W_CELL}_w_dose=${W_DOSE}_w_smooth=${W_SMOOTH}_seed${seed}"
  done
done

# ── 1. Random policy baseline (single run — obs mode agnostic) ──
echo "============================================================"
echo "  RANDOM BASELINE  seed=${SEEDS[0]}  observation_mode=${OBS_MODES[0]}"
echo "  Hyperparameters: w_cell=${W_CELL} w_dose=${W_DOSE} w_smooth=${W_SMOOTH}"
echo "============================================================"
python custom_modules/physigym/physigym/envs/run.py \
  --mode           random          \
  --seed           "${SEEDS[0]}"   \
  --observation_mode "${OBS_MODES[0]}" \
  --action_mode    targeted        \
  --w_cell         ${W_CELL}       \
  --w_dose         ${W_DOSE}       \
  --w_smooth       ${W_SMOOTH}     \
  --action_repeat  6               \
  --delta_x        0.25            \
  --delta_y        0.25            \
  --delta_radius   0.03            \
  --total_timesteps 100000         \
  --wandb          true            \
  --name           "best_hyperparameters_RANDOM_baseline_w_cell=${W_CELL}_w_dose=${W_DOSE}_w_smooth=${W_SMOOTH}_seed${SEEDS[0]}"
