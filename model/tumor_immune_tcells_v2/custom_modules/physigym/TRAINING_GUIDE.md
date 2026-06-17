# SAC Training Guide — PhysiGym TME

## Overview

This guide explains how to launch the RL agent (`run.py`), what each hyperparameter controls, and why the reward is shaped the way it is. It also documents the action constraints and frame-skip mechanism added for publication.

---

## Quickstart

```bash
cd /home/alex/Physi/PhysiCell

# Standard SAC training (full action mode — dose only, centered delivery)
python custom_modules/physigym/physigym/envs/run.py \
  --action_mode full \
  --w_cell 0.3 \
  --total_timesteps 500000 \
  --wandb true

# Targeted delivery with spatial constraints (publication config)
python custom_modules/physigym/physigym/envs/run.py \
  --action_mode targeted \
  --w_cell 0.3 \
  --w_smooth 0.02 \
  --action_repeat 4 \
  --delta_x 0.15 \
  --delta_y 0.15 \
  --delta_radius 0.20 \
  --total_timesteps 500000 \
  --wandb true

# Random policy baseline (no learning — for comparison)
python custom_modules/physigym/physigym/envs/run.py \
  --mode random \
  --action_mode targeted \
  --total_timesteps 100000 \
  --wandb false
```

---

## Reward Function

```
reward_t = w_cell × r_cancer_cells_t  −  dose_spent_t  −  w_smooth × ‖a_t − a_{t−1}‖²
```

| Term | Sign | Description |
|---|---|---|
| `w_cell × r_cancer_cells` | **+** | Reward for reducing tumor burden. `r_cancer_cells` comes from the PhysiCell env and reflects how many tumor cells died this step, normalised by `normalization_factor`. |
| `dose_spent` | **−** | Cost of administering drug this step. Directly equals the dose action value — acts as a continuous toxicity penalty. |
| `w_smooth × ‖Δa‖²` | **−** | Regularisation penalty for abrupt action changes. Keeps the policy smooth across time. Zero on the first step of each episode. |

### Why this shape?

The agent faces a **dose-efficacy tradeoff**: it must learn to dose precisely when and where it will kill tumor cells, rather than flooding the tissue continuously. A policy that never doses gets zero drug cost but also zero tumor kill. A policy that always doses at maximum gets punished by large negative `dose_spent`. The optimal policy applies drug selectively — high dose when tumors are dense, low dose otherwise.

### ⚠ Reward scale warning

From baseline tests, the random policy achieves approximately **−28 cumulative return** per episode with ~3.5 total dose and ~70 remaining tumor cells. The full-dose baseline (dose=1 every step) achieves approximately **−109** — confirming that over-dosing is correctly penalised.

**Risk of zero-action collapse:** if `w_cell` is too small relative to `dose_spent`, the agent learns to never dose (zero drug cost = best achievable return under a bad value function). To avoid this:

- Start with `--w_cell 0.3` (tested)
- If the agent converges to zero dose, increase to `--w_cell 0.5` or `1.0`
- Monitor `charts/train_action_delta_mean` in WandB — it should stay > 0

### Reward calibration sweep (publication ablation)

| `w_cell` | Expected behaviour |
|---|---|
| 0.1 | Agent likely collapses to zero dose |
| 0.3 | Balanced — recommended starting point |
| 0.5 | Agent more aggressive, may over-dose |
| 1.0 | Tumor kill dominates; dose cost is secondary |

---

## Action Space

### `action_mode=full` (default, simpler)

| Component | Range | Description |
|---|---|---|
| `drug_1_dose` | [0, 1] | Amount of drug to deliver this step |

Drug is always delivered at the center of the simulation domain with maximum radius. Spatial position is fixed — the agent only learns *when* to dose.

### `action_mode=targeted` (publication config)

| Component | Range | Description |
|---|---|---|
| `drug_1_dose` | [0, 1] | Amount of drug |
| `drug_1_x` | [0, 1] | Normalised x position (0=left, 1=right) |
| `drug_1_y` | [0, 1] | Normalised y position (0=bottom, 1=top) |
| `drug_1_radius` | [0, 1] | Normalised radius (1=maximum diagonal) |

The agent learns *where*, *how large*, and *how much* to dose simultaneously.

---

## Action Constraints (delta clipping)

For targeted mode, each action component is hard-clipped to a maximum per-step change:

```
a_t^clipped = clip(a_t^raw,  a_{t−1} − Δmax,  a_{t−1} + Δmax)
a_t^clipped = clip(a_t^clipped,  action_space.low,  action_space.high)
```

This is applied **inside the wrapper before the env receives the action**, so the clipped action is what the PhysiCell simulator sees, what gets stored in the replay buffer, and what the Q-function is trained on. The agent never observes the clipping — it must learn smooth policies organically.

At episode start, the action history is seeded with the midpoint `[0.5, 0.5, 0.5, 0.5]` so the very first action is also constrained.

### Publication values

| Argument | Value | Biological justification |
|---|---|---|
| `--delta_dose` | 1.0 (unconstrained) | Dose can change freely each decision |
| `--delta_x` | 0.15 | Drug delivery site moves ≤15% of domain per step |
| `--delta_y` | 0.15 | Idem |
| `--delta_radius` | 0.20 | Diffusion zone grows/shrinks ≤20% of max radius |

---

## Action Repeat (Frame Skip)

```
--action_repeat 4
```

The same action is applied for **4 consecutive PhysiCell steps**, and the rewards are accumulated. Only the final `next_obs` is used as the transition target in the replay buffer. This is identical to Atari's frame-skip.

**Why 4?** With `dt_gym = 15 min`, one decision step covers `4 × 15 = 60 min` of simulated time — one clinical hour. Drug dosing decisions at hourly resolution is biologically realistic. It also:

- Reduces the effective horizon, stabilising value estimation
- Increases temporal correlation of executed actions (autocorr_lag1 ≈ 0.85–0.90 observed)
- Reduces the number of gradient updates needed per episode

If any environment in the vectorised pool terminates mid-repeat, the repeat stops early for that environment.

---

## SAC Hyperparameters

| Argument | Default | Notes |
|---|---|---|
| `--total_timesteps` | 500 000 | Total env steps collected |
| `--learning_starts` | 5 000 | Steps before gradient updates begin |
| `--buffer_size` | 200 000 | Replay buffer capacity |
| `--batch_size_multiplier` | 64 | `batch_size = multiplier × num_envs` |
| `--grad_utd` | 1.0 | Gradient steps per new env step (update-to-data ratio) |
| `--num_loops` | 3 | Max gradient updates per outer loop iteration |
| `--policy_frequency` | 2 | Actor updated every N critic updates |
| `--target_network_frequency` | 1 | Target network soft-updated every step |
| `--tau` | 0.005 | Soft update coefficient |
| `--gamma` | 0.99 | Discount factor |
| `--q_lr` / `--policy_lr` | 3e-4 | Adam learning rates |
| `--autotune` | True | Automatic entropy coefficient tuning |
| `--num_envs` | 13 | Parallel environments |

### Entropy tuning (`autotune=True`)

SAC automatically adjusts the entropy coefficient `α` to match `target_entropy = −dim(action_space)`. This is critical here because the action space has 1 or 4 dimensions depending on `action_mode`. With `full` mode, target entropy = −1; with `targeted`, target entropy = −4. Monitor `charts/alpha` in WandB — it should decrease as the policy becomes more deterministic.

---

## Training Architecture

```
Actor process (CPU)                    Learner process (GPU)
──────────────────                     ─────────────────────
vec_envs (N parallel PhysiCell)        ReplayBuffer
  → collect transitions                  ← drain thread (background)
  → push to sample_queue               SAC gradient updates
  ← pull policy from actor_queue         → push policy to actor_queue
```

The drain thread continuously empties `sample_queue` into the replay buffer so the GPU is never blocked waiting for data and the actor is never blocked waiting for the queue to drain.

### Policy sync frequency

- First 1 000 grad steps: actor updated every 16 gradient steps (fast early exploration)
- After 1 000 grad steps: actor updated every 64 gradient steps

---

## Monitoring

All metrics are logged to WandB (or TensorBoard if `--wandb false`).

| Metric | What to watch for |
|---|---|
| `charts/train_return_mean50` | Should increase during training |
| `charts/test_return_mean50` | Should track train return (no large gap) |
| `charts/alpha` | Should decrease then stabilise |
| `charts/train_action_delta_mean` | Must stay > 0 (zero = collapsed policy) |
| `charts/train_action_autocorr_lag1` | Should be high (0.7+) with action_repeat=4 |
| `charts/qf1_loss` / `qf2_loss` | Should decrease and stabilise |

### Checkpoints

Saved every `--checkpoint_frequency` gradient steps (default 50 000) plus a `sac_final.pt` on exit. Resume with:

```bash
python run.py ... --resume data/<run_name>/checkpoints/sac_final.pt
```

---

## Baseline Comparison (from validation run)

Results from `test_hyperparams.py`, 3 episodes × 96 env steps (24 decision steps, action_repeat=4):

| Policy | Mean return | Dose total | Tumor remaining |
|---|---|---|---|
| Random targeted (delta-clipped) | −28.9 ± 2.6 | ~3.5 | ~70 |
| Full dose = 1.0 (always max) | −109.4 ± 7.3 | 24.0 | ~73 |

A well-trained RL agent should significantly outperform both: higher tumor kill than random, much lower dose cost than full-dose.
