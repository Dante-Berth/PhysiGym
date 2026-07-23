#!/usr/bin/env python
"""Framework overhead / throughput micro-benchmark for the PhysiGym paper (Item 2).

Times a single PhysiGym env's step() to characterise the PhysiCell<->Python<->
Gymnasium bridge cost. Run from the vroom repo root:

  ENV=custom_modules/physigym/physigym/envs
  PYTHONPATH=$ENV:. .venv/bin/python bench_framework.py

Reports median wall-clock per env.step() (which internally runs the C++
physicell.step + observation read-back), for a representative scalar mode and a
representative image mode, so the paper can quote a per-decision cost and the
image-vs-scalar observation overhead. Timing is reproducible even though rollout
*state* is not (see CLAUDE.md); we discard warm-up and report the median.
"""
import time, statistics, sys
import numpy as np
import gymnasium
import physigym  # registers physigym/ModelPhysiCellEnv-v0

SETTING = "config/PhysiCell_settings.xml"
N_STEPS = 60
N_WARMUP = 5


def bench(observation_mode):
    env = gymnasium.make(
        "physigym/ModelPhysiCellEnv-v0",
        settingxml=SETTING,
        observation_mode=observation_mode,
        render_mode=None,
    )
    env.reset(seed=42)
    a = env.action_space.sample()
    dt = []
    for i in range(N_STEPS + N_WARMUP):
        t0 = time.perf_counter()
        _, _, term, trunc, _ = env.step(a)
        dt.append(time.perf_counter() - t0)
        if term or trunc:
            env.reset(seed=42)
    env.close()
    dt = dt[N_WARMUP:]
    return statistics.median(dt), np.percentile(dt, 90)


if __name__ == "__main__":
    # One env per process (runtime allows only one PhysiCellEnv per interpreter),
    # so pass the mode as argv and invoke this script once per mode.
    mode = sys.argv[1] if len(sys.argv) > 1 else "scalars_macrophages"
    med, p90 = bench(mode)
    print(f"RESULT\t{mode}\t{med*1e3:.2f}\t{p90*1e3:.2f}\t{1.0/med:.1f}")
