#####
# title: run_physigym_tibbue_episodes.py
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
#   2. python3 custom_modules/physigym/physigym/envs/run_physigym_tibbue_episodes.py
#
# description:
#   python script to run multiple episodes from the physigym tibbue model.
#####


# library
import argparse
from extending import physicell
import gymnasium
import numpy as np
import physigym
from random import randrange


# function
def run(
        s_settingxml="config/PhysiCell_settings.xml",
        r_maxtime=1440.0,
        i_thread=8,
        i_seed=None,
    ):
    # load PhysiCell Gymnasium environment
    # %matplotlib
    # env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", settingxml="config/PhysiCell_settings.xml", cell_type_cmap="turbo", figsize=(8,6), render_mode="human", render_fps=10, verbose=True, **kwargs)
    env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", settingxml=s_settingxml)

    # episode loop
    for i_episode in range(3):

        # manipulate setting xml before reset
        env.get_wrapper_attr("x_root").xpath("//overall/max_time")[0].text = str(r_maxtime)
        env.get_wrapper_attr("x_root").xpath("//parallel/omp_num_threads")[0].text = str(i_thread)
        env.get_wrapper_attr("x_root").xpath("//save/folder")[0].text = f"output/episode{str(i_episode).zfill(8)}"

        # reset the environment
        r_reward = 0.0
        o_observation, d_info = env.reset(seed=i_seed)

        # time step loop
        b_episode_over = False
        while not b_episode_over:

            # policy according to o_observation
            d_action = {"drug_1": np.array([0.5], dtype=np.float16)}
            print("observation:", sorted(o_observation))
            print("reward:", r_reward)
            print("info:", sorted(d_info))
            print("action:", sorted(d_action))

            # action
            o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
            b_episode_over = b_terminated or b_truncated
            print("episode_over:", b_episode_over)

    # drop the environment
    env.close()


# run
if __name__ == "__main__":
    print(f"run physigym episodes ...")

    # argv
    parser = argparse.ArgumentParser(
        prog=f"run physigym episodes",
        description=f"script to run physigym episodes.",
    )
    # settingxml file
    parser.add_argument(
        "settingxml",
        # type = str,
        nargs="?",
        default="config/PhysiCell_settings.xml",
        help="path/to/settings.xml file.",
    )
    # max_time
    parser.add_argument(
        "-m",
        "--max_time",
        type=float,
        nargs="?",
        default=1440.0,
        help="set overall max_time in min in the settings.xml file.",
    )
    # thread
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        nargs="?",
        default=8,
        help="set parallel omp_num_threads in the settings.xml file.",
    )
    # seed
    parser.add_argument(
        "-s",
        "--seed",
        # type = int,
        nargs="?",
        default="none",
        help="set options random_seed in the settings.xml file and python.",
    )

    # parse arguments
    args = parser.parse_args()
    # print(args)

    # processing
    run(
        s_settingxml=args.settingxml,
        i_thread=args.thread,
        i_seed=None if args.seed.lower() == "none" else int(args.seed),
    )
