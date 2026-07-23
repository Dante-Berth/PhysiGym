# EWRL-2026 PhysiGym paper — revision status & next-session roadmap

_Last updated: 2026-07-23. This is a handoff doc so work isn't lost between sessions._

---

## ⚠️ 2026-07-23 — PAPER-GAP AUDIT (non-heuristic). Stage 6 opened.

A read-through of `physigym_ewrl2026.tex` (independent of the Stage 4/5 heuristic
work) surfaced what the paper is **missing as a framework paper**. The RL
generalization story (§3–4, image-vs-scalar OOD) is the strongest, most complete
part; the gap is that the **primary stated contribution — PhysiGym itself — is
asserted in prose but never shown**. Findings, ordered by leverage:

**Highest leverage (directly back the stated primary contribution):**
1. **No framework dataflow/architecture figure.** Every figure is about TME
   results; none depicts the PhysiCell↔Python↔Gymnasium bridge
   (`physicell_start/step/stop` + `get_cell/get_microenv/get_graph` loop). This is
   the missing centerpiece figure for a "we built a framework" paper.
2. **No minimal code snippet.** The selling point is "turn any PhysiCell model into
   an RL env with minimal code changes" / subclass `CorePhysiCellEnv`, but the
   reader never sees what that looks like. Want a ~15-line listing (obs/action/reward).
3. **No throughput/scaling measurement.** Abstract says "scalable"; §A says
   throughput is CPU-worker-limited — but nothing is measured. Need a steps/sec or
   envs-vs-throughput table + Python-extension overhead to substantiate it.

**Medium (harden the science):**
4. **Reproducibility caveat is undisclosed in the .tex.** Abstract+Conclusion claim
   "reproducible", but rollouts are NOT bit-reproducible across runs even
   single-threaded (see CLAUDE.md). Scope the claim: aggregates/curves reproduce,
   individual rollouts don't — or acknowledge the FP-nonassociativity determinism gap.
5. **Single train→test direction** (rectangle→network-field). Reverse direction is
   punted to future work but is the obvious control: without it "images generalize"
   could partly be "network-field is just easier." One reverse run would harden the
   central claim. (Already listed as optional TODO #4 above — elevate it.)
6. **M1/M2-split ablation interpreted as null with no test.** "+32.4→+26.0 does not
   improve" across 5 seeds is within noise; the "redundant" reading is asserted, not
   shown. Same for "no systematic seed-variance advantage" — leans on visual CI
   non-overlap only, no effect size / explicit test.
7. **No related-work positioning of the framework** vs. other sim↔RL bridges or the
   cited RL-on-ABM work (Zade, aif2025) — why PhysiGym over rolling your own wrapper.
8. **No biological grounding / sensitivity of the toy model** for the bio half of the
   venue (honest "toy" framing, but nothing ties rules/params to literature).

**Minor text/encoding issues:**
9. **54 dropped en/em-dashes** — an encoding issue left `\S  \S` double-space gaps
   throughout (title L32; L68,143,194,277–278,329,343,507,560,569,592,634,729,…).
   Confirmed via `grep -nP '\S  \S'`. These render as run-on gaps. A pass replacing
   the intended `` (em-dash) / en-dash / colon is a quick, high-visibility fix.
10. **Reward-eq normalization unmotivated** — the $e^{\lambda\Delta t}-1$ denominator
    in Eq. (reward) is never explained.
11. **No hardware spec** (which GPU / CPU core count) despite "single workstation".

**CORRECTED (was in my first pass, verified false):** the "64×64 µm domain looks
wrong" concern is **NOT a bug** — `config/PhysiCell_settings_env0.xml` confirms
x/y_max=63, dx=1 → genuinely a 64×64 µm / 64×64-voxel domain, so images and domain
are consistent. Worth only a *note* that the network-field correlation length
$\ell_c=45$ µm spans most of the 64 µm field (by design), not a fix.

### Stage 6 — Framework-contribution buildout (NEXT, non-heuristic)
Recommended order: (1) framework dataflow TikZ figure + (2) code snippet → these two
alone convert the least-supported part of the paper into the best. Then (3) throughput
table if sim time allows, (4) reproducibility-claim scoping (text-only, cheap),
(9) the dash/encoding pass (text-only, cheap). Items 5/6/7 depend on appetite for new runs.

**IN PROGRESS 2026-07-23:** framework dataflow figure written (`fig_framework.tex`,
TikZ, `\label{fig:framework}`) + code snippet next (items 1–2).

### Stage 6 — EXECUTION ROADMAP (decisions locked 2026-07-23)

Status legend: [x] done · [~] in progress · [ ] todo · [FW] deferred to future work.

**[x] Item 1 — framework figure + code listing.** DONE this session
(`fig_framework.tex` `fig:framework`, `listing_env.tex` `lst:env`, wired into §2.1;
compiles 19pp, 0 undefined refs; committed `paper/stage6-framework-figure`).

**[ ] Item 2 — performance/overhead characterization.** NOT STARTED. Needs a real
measurement, not text. Plan:
  1. Micro-bench the bridge: median wall-clock per `physicell_step` and per
     `env.step` (action-repeat=6) for one representative mode, single env. Isolate
     Python-extension overhead = `env.step` time − raw C++ step time.
  2. Scaling: throughput (agent-decisions/s AND simulator-steps/s) vs
     `N_env` ∈ {1,4,8,13,20,28}, holding the learner fixed. Shows the CPU-worker
     bottleneck claim quantitatively → substantiates "scalable" in the abstract.
  3. Deliverable: a small table (`tab:throughput`) in Appendix A + 2–3 sentences in
     §A "Compute". Record GPU/CPU model here too (fixes Item 7 hardware-spec gap).
  Est: ~1–2 h incl. a short timed run. No training needed (rollout-only timing).
  **Gotcha:** rollouts aren't bit-reproducible (CLAUDE.md) but *timing* is fine to
  measure; report median over ≥100 steps, warm-up discarded.

**[~] Item 3 — under-powered → make it consistent at n=5 (DECISION: 5 seeds only).**
  - **DECIDED:** keep **exactly n=5 per mode**; drop extras by **first-5-by-seed-id**
    (lowest numeric seed IDs). Verified the CSVs exist to do this:
    - I1 (`img_mc_cells`): has {1,2,4,5,42,123} → **drop 123**, keep {1,2,4,5,42}.
    - I2 (`img_mc_cells_substrates`): has {1,2,3,4,5,42,123} → **drop {42,123}**,
      keep {1,2,3,4,5}.
    - All other modes already n=5 — untouched.
  - **TODO to execute:**
    (a) add a seed-allowlist (first-5-by-id) to `plot_tme_new.py` /
        `plot_tme_bootstrap.py`; regenerate `out_tme_new/` + `out_tme_bootstrap/`.
    (b) update Table 1 (`tab:results`), Table `tab:allmodes`, Table `tab:bootstrap`,
        and the return-curve figures to the new n=5 numbers.
    (c) global text pass: replace every "n=5–7" / "5–7 seeds" / "unevenly
        distributed" with "n=5 per mode" (locations: table captions L427–434,
        §Limitations L646–648, App B L927). The "uneven seeds" mitigation sentence
        in Limitations can be shortened since it no longer applies.
  - **[FW] Reverse-direction transfer** (train network-field → test rectangle):
    **DECISION: leave as future work, no new run.** Just tighten the existing
    Limitations sentence (L650–651) so it reads as a deliberate single-direction
    scope + explicit future-work item, not an omission. Text-only.

**[ ] Item 4 — statistical testing beyond CIs. HOW TO IMPROVE (concrete):**
  The fix is to add explicit tests/effect sizes so claims don't rest on eyeballing
  CI overlap. Cheap, no new runs (uses the same per-seed end-of-training returns):
  1. **Image-vs-scalar generalization gap** (main claim): pool the 3 image modes'
     test returns vs the 4 spatial-scalar modes' test returns and report a
     **Mann–Whitney U** (rank-based, robust at small n) OR a permutation test on the
     difference of means, with a **Cliff's δ** or **Hedges' g** effect size + its CI.
     This replaces "CIs don't overlap" with a p-value + effect size.
  2. **M1/M2-split ablation** (currently asserted null): for each pair
     (I1 vs I1m, I2 vs I2m) report the **paired-by-seed difference**, its bootstrap
     CI, and a **TOST equivalence test** against a pre-stated margin (e.g. ±1σ ≈ ±10
     return). "Redundant" is only defensible as *statistical equivalence*, not a
     non-significant difference — TOST states it correctly. If TOST can't confirm
     equivalence at n=5, soften the wording to "no detectable improvement" instead of
     "redundant".
  3. **Seed-variance claim** ("no systematic advantage"): report **Levene's test** on
     the across-seed test-return spread between families; if n.s., that *is* the
     evidence for the "no systematic variance advantage" sentence.
  4. Deliverable: a short "Statistical tests" paragraph in App B + one column/row in
     `tab:bootstrap`, and reword the 3 affected claims (§Key findings L503–547,
     §M1/M2 L549–562) to cite the test + effect size. Est: ~1 h, script in
     `figures_plotting/` reusing the bootstrap seed arrays.

**[ ] Item 7 — minor text/encoding fixes (all cheap, no runs):**
  - **[FALSE — no fix]** "domain size 64µm vs 45µm kernel": VERIFIED consistent
    against `config/PhysiCell_settings_env0.xml` (x/y_max=63, dx=1 → 64µm/64vox).
    Only optionally add a note that $\ell_c=45$µm spans most of the field *by design*.
  - **[ ] Dropped en/em-dashes:** 54 `\S  \S` gaps (grep-confirmed) incl. title L32.
    These are the intended em-dash "" (or " -- ") lost in an encoding pass. Do a
    reviewed replace — NOT blind sed, since some double-spaces may be legitimate
    (post-period). Target list in Item 7 verbatim text above.
  - **[ ] Reward normalization:** add one sentence after Eq.(reward) motivating the
    $e^{\lambda\Delta t}-1$ denominator (it normalizes the observed drop by the drop
    expected from one step of intrinsic exponential growth, making $r$ scale-free in
    tumor count / comparable across ICs).
  - **[ ] Hardware spec:** fill in exact GPU + CPU core count in §A "Compute"
    (capture it during the Item 2 bench run — do these together).

**Suggested execution order** (cheap→expensive, text-first so the paper is always
compilable): Item 7 dashes+reward+hardware-stub → Item 3 text pass → Item 4 stats
script+text → Item 3 (a/b) seed-trim regen → Item 2 bench (captures hardware for 7).

---

### Stage 6 — full audit text (verbatim, as reviewed 2026-07-23)

_Note: point 7's "domain size error" is the one item verified FALSE — see the
CORRECTED note above; `config/PhysiCell_settings_env0.xml` has x/y_max=63, dx=1, so
64×64 µm / 64 voxels is consistent. The $\ell_c=45$ µm point stands only as a "spans
most of the field by design" note. Everything else below holds._

**What's lacking in the PhysiGym EWRL 2026 paper**

**1. The framework contribution is asserted, never shown.** The paper's stated primary
contribution is PhysiGym itself (Intro (1), Discussion, Conclusion), yet Section 2.1
describes it in prose only.
- No code listing / minimal example showing the "minimal code changes" claim. The whole
  selling point is "turn any PhysiCell model into an RL env with minimal code changes" —
  but a reader never sees what subclassing `CorePhysiCellEnv` actually looks like. A ~15-line
  snippet (observation/action/reward) would make the contribution concrete instead of a claim.
- No architecture/dataflow figure for the framework. Every figure is about the TME results;
  none depicts the PhysiCell↔Python↔Gymnasium bridge (the `physicell_start/step/stop` +
  `get_cell/get_microenv/get_graph` loop). For a "we built a framework" paper, this is the
  missing centerpiece figure.
- No positioning against related tools. It cites prior RL-on-ABM work (Zade, aif2025) but
  never contrasts PhysiGym with them or with other sim↔RL bridges as a framework. Why is this
  better/different than rolling your own wrapper?

**2. No performance/overhead characterization of the framework.** For an infrastructure
paper, there's no quantification of the bridge cost: steps/sec, wall-clock per simulator step,
GPU-learner vs CPU-worker throughput split, or the Python-extension overhead. "~5h/run" and
"throughput limited by CPU workers" is stated but never measured. A scaling table (envs vs
throughput) would substantiate the "scalable" claim in the abstract.

**3. The RL result is under-powered and the paper says so but doesn't fix it.**
- n=5–7 seeds, uneven across modes, self-flagged as a limitation. The bootstrap CIs mitigate
  but don't cure it — several conclusions rest on non-overlap of CIs from 5 seeds.
- Single train→test direction (rectangle→network-field). Reverse direction is punted to future
  work, but it's the obvious control: without it, "images generalize" could partly be
  "network-field is just easier." One reverse-direction run would substantially harden the
  central claim.
- Single algorithm (SAC). Fine to acknowledge, but the observation-space conclusion is
  entangled with SAC+IMPALA-CNN specifics.

**4. Weak/absent statistical testing beyond CIs.** Claims like "no systematic seed-variance
advantage" and "image vs scalar CIs disjoint" lean entirely on visual CI non-overlap. There's
no explicit significance test (e.g., a per-mode comparison or effect size), and the M1/M2-split
ablation ("+32.4→+26.0 does not improve") is interpreted as null with no test — a 6-point drop
across 5 seeds is well within noise, so the "redundant" reading is asserted, not shown.

**5. Reproducibility gaps not disclosed in the paper.** The CLAUDE.md records that rollouts are
not bit-reproducible across runs even single-threaded. That's a real caveat for a framework
paper claiming "reproducible" (abstract, conclusion) — it's currently unstated in the .tex.
Either the reproducibility claim should be scoped (curves/aggregates reproduce; individual
rollouts don't) or the determinism issue acknowledged.

**6. Toy-model realism / no biological validation.** The model is explicitly a "toy," which is
honest, but the paper makes no attempt to tie any rule/parameter to literature values, nor any
sensitivity analysis over the model's rules. For the bio audience (half the target venue),
there's nothing connecting the learned policy to a plausible biological interpretation beyond
the mechanism cartoon.

**7. Minor but concrete text issues.**
- ~~Domain size error/ambiguity~~ **VERIFIED FALSE (see note above):** the 64×64 µm domain is
  consistent with the XML and the 64×64 images; only worth a note that $\ell_c=45$ µm spans
  most of the field by design.
- Title has a missing em-dash/colon (line 31–32: "Reinforcement Learning  A State-Space Study"
  — double space where punctuation dropped). Same "  " gap recurs throughout (lines 68, 143,
  194, 329, 343, 507, 560, 569, 592, 634, 729...) — dropped en/em-dashes from an encoding issue
  that render as run-on gaps. **Confirmed: 54 occurrences via `grep -nP '\S  \S'`.**
- Reward eq. defines $\lambda$ and $\Delta t$ but the exponential-normalization denominator is
  never motivated (why $e^{\lambda\Delta t}-1$).
- No compute/hardware spec (which GPU, CPU core count) despite "single workstation."

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
