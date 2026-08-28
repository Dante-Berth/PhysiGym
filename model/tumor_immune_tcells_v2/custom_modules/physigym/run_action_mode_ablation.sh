#!/usr/bin/env bash
# Action-space ablation, matched to the `targeted` sweep on every controllable
# hyperparameter.  The point is that action_mode is the ONLY intended variable.
#
# WHY THIS EXISTS -------------------------------------------------------------
# The existing action_mode=full sweeps (wandb projects
# ..._TRAIN_RECTANGLE_TEST_NETWORKFIELD_NEW_CHEMO and
# ..._TRAIN_NETWORK_FIELD_TEST_RECTANGLE_full) cannot answer Q1, because three
# things changed alongside the action space:
#
#     key                       targeted        full (existing)
#     num_envs                  13              28
#     batch_size                832             1792      <- coupled, see below
#     wrapper.action_delta_max  [1,.25,.25,.03] None
#
# In those runs every one of the nine observation modes converges to the same
# train return (-115.2 +/- 2.0, spread 7.1) including the blind POMDP baseline,
# versus +32.8 / spread 71.0 for targeted.  Observation content buying nothing
# at all is not a believable action-space effect; it is the signature of a
# degenerate or reward-dominated policy.  See RESEARCH_LOG and
# figures_plotting/out_action_mode_ablation/.
#
# COUPLING: batch_size is NOT a flag.  run.py line ~1443 computes
#     batch_size = batch_size_multiplier * num_envs
# so 13 envs x 64 = 832 (targeted) and 28 x 64 = 1792 (full).  Matching the
# targeted sweep therefore means num_envs=13 AND batch_size_multiplier=64;
# setting one without the other silently changes the batch size.
#
# THE ONE CONFOUND THAT CANNOT BE REMOVED: run.py gates action_delta_max behind
# `action_mode == "targeted"` (line ~1417), so --delta_* is ignored here and the
# full action space is necessarily unrate-limited.  That is a property of the
# action space itself, not a config choice, and the write-up must say so rather
# than pretend the comparison is perfectly clean.
#
# Mirrors run.sh in structure; SEEDS/OBS_MODES widened to the sweep the tables use.

set -euo pipefail

SEEDS=(1 2 3 4 5)

OBS_MODES=(
  scalars_macrophages
  spatial_scalars_cells_m1m2
  spatial_scalars_cells_substrates
  spatial_scalars_cells_substrates_m1m2
  spatial_scalars_cells_spatial_no_scalars_substrates_m1m2
  img_mc_cells
  img_mc_cells_m1m2
  img_mc_cells_substrates
  img_mc_cells_substrates_m1m2
)

# ── matched to the targeted sweep ────────────────────────────────────────────
W_CELL=0.3
W_DOSE=2.0
W_SMOOTH=0.0
NUM_ENVS=13              # targeted sweep value (full sweep used 28)
BATCH_MULT=64            # 64 * 13 = 832, the targeted sweep's batch_size
ACTION_REPEAT=6
TOTAL_TIMESTEPS=100000

PROJECT=SAC_ACTION_MODE_ABLATION_MATCHED

# Both transfer directions, so the ablation covers the same cells as
# Tables 5.1 / 5.3.  Drop the second entry to run rect2net only.
DIRECTIONS=(
  "rectangle:network_field"
  "network_field:rectangle"
)

for dirpair in "${DIRECTIONS[@]}"; do
  MODE_TRAIN="${dirpair%%:*}"
  MODE_TEST="${dirpair##*:}"
  for seed in "${SEEDS[@]}"; do
    for obs in "${OBS_MODES[@]}"; do
      echo "============================================================"
      echo "  SAC  action_mode=full (MATCHED)  seed=${seed}  obs=${obs}"
      echo "  train=${MODE_TRAIN} test=${MODE_TEST}"
      echo "  num_envs=${NUM_ENVS}  batch_size=$((BATCH_MULT * NUM_ENVS))"
      echo "============================================================"
      python custom_modules/physigym/physigym/envs/run.py \
        --seed             "${seed}"          \
        --observation_mode "${obs}"           \
        --action_mode      full               \
        --mode_train       "${MODE_TRAIN}"    \
        --mode_test        "${MODE_TEST}"     \
        --w_cell           ${W_CELL}          \
        --w_dose           ${W_DOSE}          \
        --w_smooth         ${W_SMOOTH}        \
        --action_repeat    ${ACTION_REPEAT}   \
        --num_envs         ${NUM_ENVS}        \
        --batch_size_multiplier ${BATCH_MULT} \
        --total_timesteps  ${TOTAL_TIMESTEPS} \
        --wandb            true               \
        --project          "${PROJECT}"       \
        --name             "SAC_matched_full_${obs}_seed${seed}_train_${MODE_TRAIN}_test_${MODE_TEST}"
    done
  done
done

echo "============================================================"
echo "  Done.  Download with:"
echo "    python figures_plotting/download_wandb_project_full.py \\"
echo "        --project ${PROJECT} --outdir wandb_matched_full"
echo "  then add that dir to SWEEPS in plot_action_mode_ablation.py"
echo "============================================================"
