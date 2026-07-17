# EWRL-2026 PhysiGym paper — working context

This directory holds the paper (`physigym_ewrl2026.tex`) and the revision roadmap
(`REVISION_STATUS.md`). Read `REVISION_STATUS.md` first — it has the full narrative, data
provenance, regeneration commands, and the current TODO roadmap. This file is just the
orientation/gotchas layer so a fresh session doesn't have to rediscover them.

## Two repos, easy to conflate

- **`~/PhysiCell_vroom_vroom`** — the actual simulation code (C++ PhysiCell core,
  `custom_modules/physigym/` Python wrapper/gym env, trained checkpoints under `data/`).
  This is where `eval_episode_rollout.py` must be RUN FROM (see PYTHONPATH gotcha below).
- **`~/PhysiGym`** (this repo) — the paper, figures, and plotting/eval scripts
  (`figures_plotting/`). Scripts here import from the other repo at runtime.

They are separate git repos. Don't assume a file exists in one because you saw it in the
other.

## Running eval/plotting scripts — required PYTHONPATH

`eval_episode_rollout.py` and anything that imports `vectorized.py` needs BOTH the physigym
envs dir AND the vroom repo root on PYTHONPATH, run from the vroom repo root:

```bash
cd ~/PhysiCell_vroom_vroom
ENV=custom_modules/physigym/physigym/envs
PYTHONPATH=$ENV:. .venv/bin/python ~/PhysiGym/figures_plotting/eval_episode_rollout.py --mode I2 ...
```

Missing the `:.` part fails with `ModuleNotFoundError: No module named 'custom_modules'`.
Plotting-only scripts (`plot_tme_new.py`, `plot_tme_bootstrap.py`, `plot_episode_comparison.py`)
don't need this — they just read already-generated CSVs and don't import the sim.

## PhysiCell rollouts are NOT bit-reproducible across process runs

Confirmed 2026-07-17 by deep investigation (see REVISION_STATUS.md episode-figure history if
present, or ask — this cost a full session to nail down): re-running
`eval_episode_rollout.py` with the identical checkpoint, identical IC CSV, identical seed, and
even forced single-threaded (`threads_per_env=1`) execution does **not** reproduce a
previously-saved `*_replay.csv` bit-for-bit. Cell counts diverge within a single gym step.

Ruled out as causes: RNG seeding (`SeedRandom`/`UniformRandom`, thread-local `mt19937_64`,
correctly ordered relative to `omp_set_num_threads`), `threads_per_env` XML/env-var
propagation (confirmed correct), cell iteration order (plain `std::vector` index loops, not
hash/pointer-ordered). Real bug found and fixed along the way: `eval_episode_rollout.py`'s
`build()` used to call `env.reset()` twice (once internally via `init_fn()`, once explicitly)
before `rollout()`'s own reset — removed, but did not close the gap. Leading unconfirmed
hypothesis: floating-point non-associativity (compiler/FMA/CPU-affinity-dependent rounding)
rather than a logical RNG bug.

**Practical consequence:** treat any already-saved `episode_rollouts/*.csv` as the
authoritative source for published figures/numbers. Don't regenerate them to "refresh" data
unless you're deliberately accepting new numbers and updating every caption/table that cites
the old ones. If you need new columns (e.g. `action_radius`) that an old CSV lacks, that
means a full re-derivation of the narrative numbers, not a drop-in add — surface this to the
user before doing it.

## Filename mismatch in eval_episode_rollout.py

`rollout()` always writes `<mode>_seed<ic_seed>.csv`, never `<mode>_replay.csv` — even in
`--replay_ic` mode. The `*_replay.csv` files that `plot_episode_comparison.py` actually reads
were manually renamed from a `*_seed1000.csv` output at some point. If you rerun the script,
you'll get `*_seed1000.csv` again, NOT an automatic update to `*_replay.csv` — don't assume
the plot will pick up a fresh rerun without an explicit copy/rename step (which per the note
above, you probably don't want to do anyway).

## No LaTeX toolchain by default

`~/PhysiCell_vroom_vroom` (where compiles happen) didn't have TeX installed as of
2026-07-17; user ran `sudo apt-get install -y texlive-latex-extra texlive-fonts-recommended
latexmk` plus `texlive-science` (needed for `algorithm.sty`) interactively. If compiling
fails with a missing `.sty`, it's almost always a missing texlive sub-package — identify
which apt package ships it rather than guessing workarounds.

## Where the money numbers come from

Table 1 / bootstrap CIs / return curves are built from `figures_plotting/wandb_tme_new/*.csv`
(pulled from Weights & Biases run history), NOT from local checkpoint files. The matched
episode figure (`fig_episode_comparison.pdf`) is the only figure built from actual replayed
rollouts (`episode_rollouts/*.csv`). Don't confuse "we have the wandb curve" with "we have a
usable local checkpoint" — see the Stage 0 inventory in REVISION_STATUS.md for the gap
between these two.
