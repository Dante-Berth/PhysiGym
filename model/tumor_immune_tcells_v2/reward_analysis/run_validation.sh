#!/usr/bin/env bash
# Quick-start: Run all 4 biological validation tests
# Usage: bash run_validation.sh <policy_path> [outdir]

set -e

POLICY_PATH="${1:-}"
OUTDIR="${2:-./validation_results}"

if [ -z "$POLICY_PATH" ]; then
    echo "Usage: bash run_validation.sh <policy_path> [outdir]"
    echo ""
    echo "Example:"
    echo "  bash run_validation.sh runs/recommended_hyperparameters_SAC_img_mc_cells_m1m2_seed123/actor.pt"
    exit 1
fi

# Derive config path from policy path
CONFIG_JSON="${POLICY_PATH%/actor.pt}/config.json"

if [ ! -f "$POLICY_PATH" ]; then
    echo "[ERROR] Policy not found: $POLICY_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_JSON" ]; then
    echo "[ERROR] Config not found: $CONFIG_JSON"
    echo "  Looked for: $CONFIG_JSON"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  BIOLOGICAL VALIDATION — Week 3-4 Tests"
echo "════════════════════════════════════════════════════════════════"
echo "  Policy:  $POLICY_PATH"
echo "  Config:  $CONFIG_JSON"
echo "  Output:  $OUTDIR"
echo "════════════════════════════════════════════════════════════════"

PYTHON_ENV="python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/_code" && pwd)"
ROOT="/home/alex/Physi/PhysiCell"

# Set Python path
export PYTHONPATH="${ROOT}/custom_modules/physigym/physigym/envs:${PYTHONPATH}"

# ──────────────────────────────────────────────────────────────────
# TEST 1: Spatial Targeting (2-3 hours)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "[TEST 1] Spatial Targeting Validation (2-3 hours)..."
echo "  → Does agent place drug in M1-rich regions?"

TEST1_OUTDIR="$OUTDIR/test1_spatial_targeting"
mkdir -p "$TEST1_OUTDIR"

cd "$ROOT"
$PYTHON_ENV "$SCRIPT_DIR/biological_validation.py" \
    --policy_path "$POLICY_PATH" \
    --config_json "$CONFIG_JSON" \
    --obs_modes img_mc_cells_m1m2 img_mc_cells_substrates_m1m2 \
    --n_rollout_steps 500 \
    --outdir "$TEST1_OUTDIR" \
    2>&1 | tee "$TEST1_OUTDIR/test1.log"

echo "  ✓ TEST 1 complete. Check: $TEST1_OUTDIR/fig_spatial_targeting.png"

# ──────────────────────────────────────────────────────────────────
# TEST 2: Generalization (4-6 hours)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "[TEST 2] Generalization Across Tumor Burdens (4-6 hours)..."
echo "  → Does policy work on tumor_burden ∈ {64, 128, 256}?"

TEST2_OUTDIR="$OUTDIR/test2_generalization"
mkdir -p "$TEST2_OUTDIR"

cd "$ROOT"
$PYTHON_ENV "$SCRIPT_DIR/biological_validation.py" \
    --policy_path "$POLICY_PATH" \
    --config_json "$CONFIG_JSON" \
    --obs_modes img_mc_cells_m1m2 \
    --tumor_burdens 64 128 256 \
    --n_gen_episodes 5 \
    --outdir "$TEST2_OUTDIR" \
    2>&1 | tee "$TEST2_OUTDIR/test2.log"

echo "  ✓ TEST 2 complete. Check: $TEST2_OUTDIR/fig_generalization.png"
echo "     Data: $TEST2_OUTDIR/generalization_results.csv"

# ──────────────────────────────────────────────────────────────────
# TEST 3: Immune Mechanism (1-2 hours log analysis)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "[TEST 3] Immune Mechanism Timing (1-2 hours log analysis)..."
echo "  → Does drug precede T-cell activation by ~5-7 hours?"
echo "  [This is semi-manual — see BIOLOGICAL_VALIDATION_WEEK34.md]"

TEST3_OUTDIR="$OUTDIR/test3_immune_timing"
mkdir -p "$TEST3_OUTDIR"

echo "  ⓘ Logs should be in: runs/*/episode_*.csv"
echo "  ⓘ Run extract_trajectories.py to prepare data, then analyze_immune_mechanism()"

# ──────────────────────────────────────────────────────────────────
# TEST 4: Ablation (2-3 hours)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "[TEST 4] Ablation: M1/M2 Channel Importance (2-3 hours)..."
echo "  → Reward loss when removing M1/M2 channels?"

TEST4_OUTDIR="$OUTDIR/test4_ablation"
mkdir -p "$TEST4_OUTDIR"

cd "$ROOT"
$PYTHON_ENV "$SCRIPT_DIR/biological_validation.py" \
    --policy_path "$POLICY_PATH" \
    --config_json "$CONFIG_JSON" \
    --obs_modes img_mc_cells_m1m2 \
    --n_rollout_steps 500 \
    --outdir "$TEST4_OUTDIR" \
    2>&1 | tee "$TEST4_OUTDIR/test4.log"

echo "  ✓ TEST 4 complete. Check: $TEST4_OUTDIR/"

# ──────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  VALIDATION COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Outputs:"
echo "  Test 1 (Spatial):       $TEST1_OUTDIR/"
echo "  Test 2 (Generalization): $TEST2_OUTDIR/"
echo "  Test 3 (Immune timing):  $TEST3_OUTDIR/ (manual setup)"
echo "  Test 4 (Ablation):       $TEST4_OUTDIR/"
echo ""
echo "Next: Read BIOLOGICAL_VALIDATION_WEEK34.md for interpretation guide"
echo "      Copy results into Section 4.3 of your thesis"
echo ""
echo "════════════════════════════════════════════════════════════════"
