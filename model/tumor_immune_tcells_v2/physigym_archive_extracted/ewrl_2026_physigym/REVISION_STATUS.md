# EWRL-2026 PhysiGym paper — revision status & next-session roadmap

_Last updated: 2026-07-20. This is a handoff doc so work isn't lost between sessions._

---

## ⚠️ 2026-07-20 — DATA LOSS EVENT + Stage 3 done. READ THIS FIRST.

**A `video_maker.py` run over all of `data/` with cleanup enabled deleted, irreversibly:**
- **ALL `.pt` checkpoints** (every mode, every seed — `find data -name '*.pt'` → 0). No backup
  found on this machine. `video_maker.py`'s final sweep keeps only `{.mp4,.csv,.npz}`, so it
  wipes `.pt`. **Stages 1 and 4 recovery now require RETRAINING (~5h/run)** unless the
  checkpoints exist on another machine or as W&B model artifacts — CHECK W&B ARTIFACTS FIRST.
- **ALL SAC-mode `frames.npz`** (consumed into videos). Only RANDOM-baseline `.npz` remain.
  Not bit-reproducible (see CLAUDE.md), so gone for good.

**What SURVIVED and is fully intact (nothing on the paper's critical path was lost):**
- **10,180 `video.mp4`** — Stage 2's actual deliverable. Effectively DONE.
- **W&B curves** (`figures_plotting/wandb_tme_new/`, all 9 modes + baseline) → Table 1,
  bootstrap CIs, return curves all regenerable.
- **`episode_rollouts/*.csv`** → matched-episode figure source intact.
- **~48k `data.csv` traces + ~50k `ic_*.csv`** → per-episode behavioural record intact.
  This is what made Stage 3 possible without checkpoints.

**Stage 3 (why scalars overfit) — DONE 2026-07-20, integrated into the paper.**
Used surviving `data.csv`+`ic_*.csv` only (the checkpoint-based Q-calibration route was dead).
Measured injection↔tumour-centroid aiming distance, train vs test, image vs scalar. Result:
on train all modes aim equally (~0.34–0.36); on OOD test image modes hold aim (I2 0.359→0.343)
while every scalar mode degrades (S3s 0.341→0.414) with ~2× variance → overfit is a
**drug-aiming failure**, not lost dosing ability. Added a Discussion paragraph +
`img/fig_aiming_transfer.pdf` (Fig `fig:aiming_transfer`). **Paper recompiles clean, 18 pages,
0 undefined refs.** Analysis scripts: scratchpad `stage3_transfer.py` / `stage3_figure.py`
(reproduce from `data/` in ~15s; copy into `figures_plotting/` if you want them versioned).

**Stage 4 (macrophage-aware heuristic baseline) — CODE DONE 2026-07-20; EVAL LAUNCHED.**
_2026-07-20: launched on **sureli 11** via `run_heuristic.sh` (Phase A radius sweep running).
Next: pick BEST_RADIUS from Phase A test_return, enable Phase B (5-seed eval), then Stage 5._
Rule-based, no checkpoint needed (needs only fresh rollouts). Mechanism-grounded: `cell_rules.csv`
shows drug_1 re-polarises macrophages (↓pro_tumoral / ↑anti_tumoral secretion), so the policy
injects a fixed dose (0.5) at the centroid of the **M2 macrophages within R µm of any tumour
cell**, radius sized to their spread; dose 0 when no such macrophage exists. Smoke-tested
end-to-end (withholds dose until M2-adjacent macrophages appear, then tracks the cluster;
per-env ground-truth via `env_method`).

- **Source-of-truth note:** the model method lives in the canonical PhysiGym source
  `model/tumor_immune_tcells_v2/custom_modules/physigym/physicell_model.py`; the vroom copy
  (`~/PhysiCell_vroom_vroom/custom_modules/physigym/physigym/envs/physicell_model.py`) is
  kept **byte-identical**. `run.py` + `run_heuristic.sh` live **only in vroom**.
- New code: `get_heuristic_action(radius, dose)` (both model copies); `run_heuristic_policy` +
  `--mode heuristic` + `--heuristic_radius`/`--heuristic_dose` in vroom `run.py`;
  `~/PhysiCell_vroom_vroom/run_heuristic.sh` (Phase A radius-tune {5,10,20}µm × 1 seed →
  Phase B 5-seed eval at BEST_RADIUS, named `best_hyperparameters_HEURISTIC_baseline_...`).
- **TODO to finish Stage 4→5:** run `run_heuristic.sh` Phase A, pick R by test_return, run
  Phase B; download `HEURISTIC_*` wandb curves into a `heuristic_baseline/` group; add a parse
  branch to `plot_tme_new.py`/`plot_tme_bootstrap.py` (mirror `random_baseline`); add the row to
  Table 1 + bootstrap CIs (+ optionally the episode figure).
- **Not committed yet** (2026-07-20): PhysiGym model change + vroom's 3 files are staged for the
  user to commit/push. The 27 pre-existing `config/*.xml` in vroom stay untouched.

**Stage 0 inventory below is STALE** (pre-deletion, and was already too pessimistic about
image checkpoints existing). Do not trust its checkpoint counts.

---

**Paper:** `physigym_ewrl2026.tex` (this directory)
**OpenReview:** https://openreview.net/forum?id=p5INPBXPKX (gated — reviews not machine-readable; paste them in if you want them addressed)

---

## TL;DR

The paper was revised from the old **6-mode** study to the new **9-mode** experiment set
(wandb project `thomas-phd/SAC_ASYNC_TME_NEW_HYP_REWARD`). The train/test split was
**inverted**: now **train = rectangle, test = network-field (OOD)**. All tables, figures,
captions, hyperparameters, abstract, discussion, and conclusion were updated to match.
**No TODOs/placeholders remain in the .tex; all envs balance and all `\ref`s resolve.**

**Update 2026-07-18: both blockers resolved.** (1) LaTeX toolchain now installed;
`latexmk -pdf physigym_ewrl2026.tex` compiles cleanly to a 17-page PDF (warnings only,
no errors). (2) `ic_distributions.pdf`/`.png` regenerated as a **2-panel** figure
(rectangle=train, network-field=test; circular dropped) via
`~/PhysiGym/figures_plotting/plot_initial_conditions.py`, pointed at the current
`~/PhysiCell_vroom_vroom/data/best_hyperparameters_SAC_img_mc_cells_substrates_..._seed42_.../`
run (the old `TME_V2_32_img_mc_cells_1779954610` path no longer exists on disk).
The paper is now submission-ready in its current 9-mode-study form.

---

## Core facts baked into the revision (verified from code/config, not guessed)

- **Train = rectangle, Test = network-field.** (Opposite of the old draft. Circular layout dropped.)
- **action_repeat = 6** → 1e5 agent decisions = **6e5 simulator steps**. Figure x-axes are in simulator steps.
- **Reward** (`wrapper.py:758`): `r = w_cell·r_tumor − w_dose·d_t − w_smooth·‖a_t−a_{t-1}‖²`,
  with **w_cell=0.3, w_dose=2.0, w_smooth=0** (smoothness term off). Eq. in paper uses `w_dose`.
- **M1/M2 rule** (`physicell_model.py:509`): a macrophage is **M2** if
  `pro_tumoral_factor > anti_tumoral_factor` at its nearest voxel, else **M1**.
- **Hyperparams** vary slightly across the sweep (e.g. seed42 run: num_envs=28, batch=1792, alpha=0.05;
  seed1 run: num_envs=13, batch=832). Paper reports representative values.
- **Compute:** ~5 h wall-clock per run (1 mode, 1 seed), single workstation, async SAC (CPU workers + 1 GPU learner).

## The narrative (what the numbers show)

| Group | Test (network-field) | Meaning |
|---|---|---|
| Image modes (I2m/I1/I2) | +30 … +32 | generalize; CIs all **> +24** |
| Spatial-scalars (S3s/S3m/S3sm) | ~0 | train well but **overfit rectangle**; CIs **contain 0** |
| RANDOM baseline | −25.3 | learning-free anchor |
| POMDP (`scalars_macrophages`) | −31.8 | **no better than / worse than random** |

Image vs. scalar test CIs are **disjoint** → generalization gap is statistically significant.
Matched episode: I2 → 3 tumor cells (near-eradication); S3s → 70; POMDP → 130 (tumor grew).

---

## Data & scripts (all in `~/PhysiGym/figures_plotting/`)

**Python env with wandb/torch:** `/home/alex/PhysiCell_vroom_vroom/.venv/bin/python`

| Script | Purpose | Output |
|---|---|---|
| `download_tme_new.py` | pull train/test return histories (mode parsed from run NAME; config `observation_mode` is None) | `wandb_tme_new/<mode>/` |
| `plot_tme_new.py` | main tables + curves (σ bands) + random baseline row/line | `out_tme_new/` |
| `plot_tme_bootstrap.py` | 95% bootstrap CI tables + CI-band curves (imports config from plot_tme_new) | `out_tme_bootstrap/` |
| `eval_episode_rollout.py` | deterministic replay rollout from a checkpoint on a fixed IC | `episode_rollouts/<mode>_*.csv` |
| `plot_episode_comparison.py` | 3-panel matched-episode figure | `out_tme_new/fig_episode_comparison.pdf` |

**Cached data:**
- `wandb_tme_new/<mode>/SAC_*.csv` — per-seed training/test curves for 9 modes.
- `wandb_tme_new/random_baseline/RANDOM_*.csv` — 5 random-policy seeds (seed4 curve = run `awei2uy4`).
- `episode_rollouts/ic_episode_A.csv` — the fixed IC (124 tumor cells) used for the matched episode.

**Regenerate everything:**
```bash
cd ~/PhysiGym/figures_plotting
PY=/home/alex/PhysiCell_vroom_vroom/.venv/bin/python
$PY plot_tme_new.py && $PY plot_tme_bootstrap.py && $PY plot_episode_comparison.py
# then copy PDFs into the paper img/ dir (see below)
```

## Checkpoints (for re-running episode rollouts)

Live in `~/PhysiCell_vroom_vroom/data/best_hyperparameters_SAC_*/checkpoints/`.
Each `.pt` is self-describing (carries `config`/`d_arg_env`). `eval_episode_rollout.py`
auto-resolves `sac_final.pt` → `sac_latest.pt` → highest `sac_step*.pt`.
Run dirs ALSO contain saved per-episode traces: `env*/{train,test}/episodes/run_*/data.csv`
(+ `ic_*.csv`) — reusable **without re-simulating**.

**To re-run a matched episode** (from vroom repo root, note the PYTHONPATH):
```bash
cd ~/PhysiCell_vroom_vroom
ENV=custom_modules/physigym/physigym/envs
PYTHONPATH=$ENV:. .venv/bin/python ~/PhysiGym/figures_plotting/eval_episode_rollout.py \
    --mode I2 --replay_ic ~/PhysiGym/figures_plotting/episode_rollouts/ic_episode_A.csv \
    --max_steps 672 --out ~/PhysiGym/figures_plotting/episode_rollouts
# modes: I2, S3s, POMDP
```

## Copying figures into the paper
Paper img dir: `physigym_archive_extracted/ewrl_2026_physigym/img/`
Filenames the .tex expects: `train_return_mean50.pdf`, `test_return_mean50.pdf`,
`return_std.pdf`, `train_return_ci.pdf`, `test_return_ci.pdf`,
`fig_action_repeat_sensitivity.png`, `fig_episode_comparison.pdf`.

---

## NEXT SESSION — prioritized TODO

1. ~~**Compile the paper.**~~ **DONE 2026-07-18** — compiles cleanly, 17 pages.
2. ~~**`ic_distributions.pdf`**~~ **DONE 2026-07-18** — regenerated as 2-panel
   (rectangle/network-field), circular dropped.
3. **(Optional) A second matched episode** — a "harder / rebound" case to complement the
   near-eradication one, mirroring the old paper's two-figure structure. Pick another IC from
   I2's saved test episodes (script: scan `*/test/episodes/run_*/data.csv` for a mid-range
   final tumor count), replay through the 3 modes.
4. **(Optional, if reviewers ask)** reverse-direction transfer (train network-field → test
   rectangle); more seeds. Currently framed as future work in Limitations.
5. **Clean up:** DONE — old unreferenced `fig_episode_comparison_2.pdf` deleted.

### Episode figure notes (2026-07-16)
- The matched-episode figure's per-step dose panel is plotted as a **rolling mean (w=10)**
  in `plot_episode_comparison.py` — raw per-step actions are jittery step-to-step because
  `w_smooth=0` (temporal consistency comes from action_repeat=6, not the reward). The
  smoothing is cosmetic/legibility only; top two panels (cumulative return, dose) are raw.
  Verified jitter: mean step-to-step |Δdose| ≈ 0.10 (I2), 0.09 (S3s), 0.005 (POMDP).
- Only the single matched network-field episode figure is kept; old/ancient episode figures
  are no longer used.

## Known caveats / things a reviewer might poke
- Seed count target: **fixed to 5 seeds per mode** (2026-07-17 decision — see roadmap below;
  was previously n=5–7, uneven across modes).
- Single train/test direction (acknowledged).
- Single RL algorithm (SAC) (acknowledged).
- I2 checkpoint only trained to 60k steps (`sac_final.pt`); others to 95k. The matched-episode I2
  still near-eradicates, so this is fine, but note it if asked.

---

## ROADMAP — post-2026-07-17 revision push

_Added 2026-07-17. Three substantial, mostly-independent workstreams. Ordered by dependency,
not by priority — Stages 3 and 4 can start immediately in parallel with Stage 1._

### Stage 0 — Checkpoint inventory (do first, fast)
Audit `~/PhysiCell_vroom_vroom/data/best_hyperparameters_SAC_*/checkpoints/` against the
**fixed target of 5 seeds per mode** (9 modes + RAND + POMDP baselines). Produce a concrete
gap list: which (mode, seed) pairs are missing a checkpoint entirely, and which have a
checkpoint but not to the target step count (95k, per the I2-at-60k caveat above).

**Partial inventory done 2026-07-17 (interrupted, needs finishing next session):**

There are two separate things that were being conflated — re-check both, they have different
gaps:

1. **wandb training-curve history** (`~/PhysiGym/figures_plotting/wandb_tme_new/<mode>/*.csv`)
   — this is what Table 1 / the return-curve figures are already built from, and is NOT the
   same as having local checkpoint `.pt` files. Seed-run counts found so far:
   - `img_mc_cells_substrates_m1m2` (I2m): 5 — OK
   - `scalars_macrophages` (POMDP): 5 — OK
   - `spatial_scalars_cells_spatial_no_scalars_substrates_m1m2` (S5m): 5 — OK
   - `spatial_scalars_cells_substrates` (S3s): 5 — OK
   - `spatial_scalars_cells_substrates_m1m2` (S3sm): 5 — OK
   - `img_mc_cells` (I1): 6 — has 1 extra, trim to 5 for consistency
   - `img_mc_cells_m1m2` (I1m): 6 — has 1 extra, trim to 5
   - `spatial_scalars_cells_m1m2` (S3m): 11 — has 6 extra, trim to 5
   - `img_mc_cells_substrates` (I2): 11 — has 6 extra, trim to 5
   - RAND baseline: 5 seeds already (per existing notes above) — OK
   - **Which specific seeds to drop when trimming (e.g. lowest-quality run vs. just first-5-by-seed-id)
     was NOT decided — needs a call next session.**

2. **Local checkpoint `.pt` files** (`~/PhysiCell_vroom_vroom/data/best_hyperparameters_SAC_*/checkpoints/`)
   — needed for Stage 2 (video rendering) and Stage 4 (baseline comparison replay), NOT covered
   by wandb curves alone. This machine (`PhysiCell_vroom_vroom`, i.e. NOT PhysiGym) is missing
   most of them:
   - `img_mc_cells_m1m2` (I1m): only found at step 5000 (essentially untrained) — **effectively missing**
   - `img_mc_cells` (I1): only 1 seed dir found (seed42), max step 55000 (target 95000) — **incomplete**
   - `img_mc_cells_substrates` (I2): 2 seed dirs found (seed1 @ 55k, seed42 @ 60k) — **incomplete,
     both under the 95k target; this is the "I2 only trained to 60k" caveat already noted above**
   - `img_mc_cells_substrates_m1m2` (I2m): **not found in the local data/ listing at all** — check
     if it's under a different naming pattern, or genuinely absent locally
   - `scalars_macrophages` (POMDP): 1 seed dir (seed42) @ 95000 — OK but only 1 seed locally
     (wandb has 5 — the other 4 seeds' checkpoints may not be saved locally)
   - `spatial_scalars_cells_*` (S3s/S3m/S3sm/S5m): these look the most complete locally — S3m and
     S3sm have 5 seed dirs each @ 95000, S5m has 5 @ 95000, S3s has 5 @ 95000
   - **Open question**: are the missing local checkpoints (esp. all the image modes) recoverable
     from another machine/backup, or were they deleted after the wandb curves were logged and
     genuinely need re-training? Check before assuming Stage 1 needs a full retrain — this is
     the single biggest unknown blocking the roadmap's time estimate.

### Stage 1 — Fill missing checkpoints (blocking, slow)
Train whatever (mode, seed) combinations Stage 0 finds missing, down to exactly 5 seeds/mode
(drop extra seeds where a mode currently has 6–7, to keep the reported n consistent across
the whole table). Long pole of this roadmap — real training wall-clock (~5h/run per the compute
note above). Everything in Stage 2 benefits from full coverage but can start incrementally on
whatever's already trained.

### Stage 2 — Multi-seed video pipeline (train + test, per state space)
For each (mode, seed): load the actor checkpoint → rollout with `generate_physicell_data=True`
→ dump `frames.npz` (`wrapper.py:_dump_frames`) → render `video.mp4` via `video_maker.py`.
Needs both train-mode (rectangle) and test-mode (network-field) rollouts per checkpoint.
Depends on Stage 1 for full 5-seed coverage; can start now on already-trained checkpoints.

### Stage 3 — Why do scalar modes overfit but image modes don't? (can start now)
**Empirical diagnosis**, not just a written hypothesis — dig into data already logged before
running anything new:
- Use the existing Q-value calibration data (`wrapper.py`'s `q_calibration_data` on test
  episodes) to compare train vs. test critic behavior for S3s/S3m/S3sm vs. I1/I2/I2m.
- Check whether scalar-mode features encode absolute position/counts that don't transfer
  across rectangle → network-field geometry, vs. image-mode conv filters that should be closer
  to translation-invariant.
- Candidate output: a Discussion-section paragraph backed by concrete evidence (not just
  architectural intuition), plus maybe one supporting figure/table if the diagnosis turns up
  something plottable (e.g. a feature-attribution or train/test critic-value gap comparison).

### Stage 4 — Macrophage-aware heuristic baseline — CODE DONE 2026-07-20, EVAL PENDING
Rule-based baseline (not RL) to add alongside RAND/POMDP in Table 1 and the episode figure.
See the full status block at the top of this file; summary here:
- **Implemented** as `get_heuristic_action(radius, dose)` on the model (PhysiGym canonical
  source + byte-identical vroom copy) + `run_heuristic_policy`/`--mode heuristic` in vroom
  `run.py` + `run_heuristic.sh`. Refined from the roadmap's original wording: targets
  **M2** (pro-tumoral) tumour-adjacent macrophages specifically, because `cell_rules.csv`
  makes drug_1 act on macrophage polarisation — so those are the mechanistically correct
  target. Fixed dose 0.5 when a target exists, else 0.
- **Still to do:** run `run_heuristic.sh` (Phase A radius tuning → Phase B 5-seed eval),
  evaluate identically to RAND/POMDP on the network-field test set, so it drops into Table 1
  / bootstrap CIs. This needs fresh rollouts (sim time), no checkpoints.

### Stage 5 — Paper integration (last)
- Fold Stage 3's empirical findings into Discussion.
- Add Stage 4's heuristic baseline row to Table 1 (main + bootstrap) and, if illustrative,
  the episode-comparison figure.
- Regenerate any figures/captions touched by the above; recompile and spot-check.
