#####
# title: run_physigym_tib.py
#
# language: python3
# library: gymnasium, numpy,
#   and the extending and physigym custom_modules
#
# date: 2024-spring
# license: bsb-3-clause
# author: Alexandre Bertin, Elmar Bucher
# input: https://gymnasium.farama.org/main/
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# run:
#   1. copy this file into the PhysiCell root folder
#   2. python3 run_physigym_tib.py
#
# description:
#   python script to run a single episode from the physigym tib model.
#####


# library
from extending import physicell
import gymnasium
import numpy as np
import physigym
from random import randrange
import random
from custom_modules.physigym.physigym.envs.run_physigym_tib_sac import (
    PhysiCellModelWrapper,
)

# load PhysiCell Gymnasium environment
# %matplotlib
# env = gymnasium.make('physigym/ModelPhysiCellEnv-v0', settingxml='config/PhysiCell_settings.xml', figsize=(8,6), render_mode='human', render_fps=10)
env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", observation_mode="scalars_cells")
env = PhysiCellModelWrapper(env)


def treatment_regime(env, treatment_value=0):
    r_reward = 0.0
    liste = []
    o_observation, d_info = env.reset()
    b_episode_over = False
    liste = []
    while not b_episode_over:
        # policy according to o_observation
        d_observation = o_observation
        treatment_value = (
            random.random() if treatment_value is None else treatment_value
        )
        d_action = np.array([treatment_value], dtype=np.float16)

        # action
        o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
        b_episode_over = b_terminated or b_truncated
        liste.append(
            [
                env.unwrapped.step_episode,
                d_info["reward_drugs"],
                d_info["reward_cancer_cells"],
                d_info["number_tumor"],
                d_action,
            ]
        )
    return liste


mylist = treatment_regime(env, treatment_value=0.0)
print(mylist)
# drop the environment
env.close()
