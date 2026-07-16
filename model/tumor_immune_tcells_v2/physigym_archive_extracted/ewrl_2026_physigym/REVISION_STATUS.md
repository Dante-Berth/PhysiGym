# EWRL-2026 PhysiGym paper — revision status & next-session roadmap

_Last updated: 2026-07-16. This is a handoff doc so work isn't lost between sessions._

**Paper:** `physigym_ewrl2026.tex` (this directory)
**OpenReview:** https://openreview.net/forum?id=p5INPBXPKX (gated — reviews not machine-readable; paste them in if you want them addressed)

---

## TL;DR

The paper was revised from the old **6-mode** study to the new **9-mode** experiment set
(wandb project `thomas-phd/SAC_ASYNC_TME_NEW_HYP_REWARD`). The train/test split was
**inverted**: now **train = rectangle, test = network-field (OOD)**. All tables, figures,
captions, hyperparameters, abstract, discussion, and conclusion were updated to match.
**No TODOs/placeholders remain in the .tex; all envs balance and all `\ref`s resolve.**

The paper is **submission-ready modulo two things**: (1) it has **not been compiled**
(no LaTeX toolchain on this machine), (2) `ic_distributions.pdf` is **stale** (see below).

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

1. **Compile the paper.** No LaTeX toolchain here. Run `latexmk -pdf physigym_ewrl2026.tex`
   (or `pdflatex` ×2 + `bibtex`). Fix any typesetting issues. **Do this first** — everything
   else is moot if it doesn't build.
2. **`ic_distributions.pdf`** — the figure FILE is stale (Jun 17, still shows old circular
   layout), but circular is no longer part of the study. Only matters if this figure renders
   in the compiled PDF: if so, regenerate to show rectangle + network-field via
   `~/PhysiGym/figures_plotting/plot_initial_conditions.py`. Otherwise ignore.
   (User decision 2026-07-16: circular no longer needed; paper considered done.)
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
- Seed count n=5–7, uneven across modes (acknowledged in Limitations; mitigated with bootstrap CIs).
- Single train/test direction (acknowledged).
- Single RL algorithm (SAC) (acknowledged).
- I2 checkpoint only trained to 60k steps (`sac_final.pt`); others to 95k. The matched-episode I2
  still near-eradicates, so this is fine, but note it if asked.
