#####
# title: run_physigym_episode.py
#
# language: python3
#
# date: 2024-spring
# license: bsb-3-clause
# author: Alexandre Bertin, Elmar Bucher
# input: https://gymnasium.farama.org/main/
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# run:
#   1. cd path/to/PhysiCell
#   2. python3 custom_modules/physigym/physigym/envs/run_physigym_episode.py
#
# description:
#   python script to run a single episode from the physigym episode model.
#####


# library
from extending import physicell
import gymnasium
import numpy as np
import physigym

# load PhysiCell Gymnasium environment
# %matplotlib
# env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", settingxml="config/PhysiCell_settings.xml", cell_type_cmap="turbo", figsize=(8,6), render_mode="human", render_fps=10, verbose=True, **kwargs)
env = gymnasium.make("physigym/ModelPhysiCellEnv-v0")

# reset the environment
r_reward = 0.0
o_observation, d_info = env.reset()

# time step loop
b_episode_over = False
while not b_episode_over:

    # policy according to o_observation
    d_observation = o_observation
    d_action = {
        "secretion_a": np.array([0.0]),
        "secretion_b": np.array([0.0]),
        "secretion_c": np.array([0.0]),
    }
    # celltype a
    if (d_observation["celltype_a"][0] <= physicell.get_parameter("cell_count_target")):
        d_action.update({"secretion_a": np.array([(1 - r_reward) * 1/12])})
    # celltype b
    if (d_observation["celltype_b"][0] <= physicell.get_parameter("cell_count_target")):
        d_action.update({"secretion_b": np.array([(1 - r_reward) * 1/12])})
    # celltype c
    if (d_observation["celltype_c"][0] <= physicell.get_parameter("cell_count_target")):
        d_action.update({"secretion_c": np.array([(1 - r_reward) * 1/12])})

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
