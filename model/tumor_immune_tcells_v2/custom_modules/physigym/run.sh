#!/usr/bin/env bash
# Launch one SAC run per observation mode, sequentially.
# Publication hyperparameters: targeted delivery, action_repeat=4, delta clipping, w_smooth.

SEEDS=(200)

OBS_MODES=(
  img_mc_cells_substrates
  img_mc_cells
  scalars_macrophages
)

# ── 1. Random policy baseline (single run — obs mode agnostic) ──
echo "============================================================"
echo "  RANDOM BASELINE  seed=${SEEDS[0]}  observation_mode=${OBS_MODES[0]}"
echo "============================================================"
python custom_modules/physigym/run.py \
  --mode           random          \
  --seed           "${SEEDS[0]}"   \
  --observation_mode "${OBS_MODES[0]}" \
  --action_mode    targeted        \
  --action_repeat  4               \
  --delta_x        0.15            \
  --delta_y        0.15            \
  --delta_radius   0.05            \
  --total_timesteps 500000         \
  --wandb          true            \
  --name           "RANDOM_baseline_seed${SEEDS[0]}"

# ── 2. SAC training runs (one per obs mode) ─────────────────────
for seed in "${SEEDS[@]}"; do
  for obs in "${OBS_MODES[@]}"; do
    echo "============================================================"
    echo "  SAC  seed=${seed}  observation_mode=${obs}"
    echo "============================================================"
    python custom_modules/physigym/run.py \
      --seed           "${seed}"  \
      --observation_mode "${obs}" \
      --action_mode    targeted   \
      --w_cell         0.3        \
      --w_smooth       0.02       \
      --action_repeat  4          \
      --delta_x        0.15       \
      --delta_y        0.15       \
      --delta_radius   0.05       \
      --total_timesteps 500000    \
      --wandb          true       \
      --name           "SAC_${obs}_seed${seed}"
  done
done
