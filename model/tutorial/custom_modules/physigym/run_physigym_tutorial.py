#####
# title: run_physigym_tutorial.py
#
# language: python3
#
# date: 2024-spring
# license: <has to be comatiple with bsb-3-clause>
# author: <your name goes here>
# input: https://gymnasium.farama.org/main/
# original source code: https://github.com/Dante-Berth/PhysiGym
# modified source code: <https://>
#
# run:
#   1. cd path/to/PhysiCell
#   2. python3 custom_modules/physigym/physigym/envs/run_physigym_tutorial.py
#
# description:
#   python script to run a single episode from the physigym tutorial model.
#####


# library
from extending import physicell
import gymnasium
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
    i_observation = o_observation[0]
    if (i_observation >= physicell.get_parameter("cell_count_target")):
        d_action = {"drug_dose": np.array([1.0 - r_reward])}
    else:
        d_action = {"drug_dose": np.array([0.0])}

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
