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

# load PhysiCell Gymnasium environment
# %matplotlib
# env = gymnasium.make('physigym/ModelPhysiCellEnv-v0', settingxml='config/PhysiCell_settings.xml', figsize=(8,6), render_mode='human', render_fps=10)
env = gymnasium.make("physigym/ModelPhysiCellEnv-v0")

# reset the environment
r_reward = 0.0
o_observation, d_info = env.reset()

# time step loop
b_episode_over = False
while not b_episode_over:
    # policy according to o_observation
    d_observation = o_observation
    d_action = {"drug_1": np.array([0.5], dtype=np.float16)}

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
