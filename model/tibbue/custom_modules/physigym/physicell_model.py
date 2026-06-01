#####
# title: pysigym/envs/physicell_model.py
#
# language: python3
# library: gymnasium v1.0.0a1
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     model specific implementation of the custom_modules/extending module
#     comaptible Gymnasium environment.
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
#####


# library
from extending import physicell
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import os
import pandas as pd
from physigym.envs.physicell_core import CorePhysiCellEnv
import skimage as ski


# function
class ModelPhysiCellEnv(CorePhysiCellEnv):
    """
    input:
        physigym.CorePhysiCellEnv

    output:
        physigym.ModelPhysiCellEnv

    run:
        import gymnasium
        import physigym

        env = gymnasium.make("physigym/ModelPhysiCellEnv")

        o_observation, info = env.reset()
        o_observation, r_reward, b_terminated, b_truncated, info = env.step(action={})
        env.close()

    description:
        this is the model physigym environment class, built on top of the
        physigym.CorePhysiCellEnv class, which is built on top of the
        gymnasium.Env class.

        fresh from the PhysiGym repo this is only a template class!
        you will have to edit this class, to specify the model specific
        reinforcement learning environment.
    """

    def __init__(
        self,
        settingxml="config/PhysiCell_settings.xml",
        cell_type_cmap="turbo",
        figsize=(9, 9),  # inch
        render_mode=None,
        render_fps=10,
        verbose=True,
        # **kwargs
        observation_mode="scalars",
        img_rgb_grid_size_y=64,  # pixel
        img_rgb_grid_size_x=64,  # pixel
        img_mc_grid_size_x=64,  # pixel
        img_mc_grid_size_y=64,  # pixel
        normalization_factor=512,
    ):
        # check redner mode
        if observation_mode == "img_rgb" and render_mode == None:
            raise ValueError(
                f"if observation_mode is img_rgb the render_mode can not be None. try: {self.metadata['render_modes']}."
            )

        # call super class init
        super().__init__(
            settingxml=settingxml,
            cell_type_cmap=cell_type_cmap,
            figsize=figsize,
            render_mode=render_mode,
            render_fps=render_fps,
            verbose=verbose,
            # **kwargs
            observation_mode=observation_mode,
            img_rgb_grid_size_x=img_mc_grid_size_x,
            img_rgb_grid_size_y=img_mc_grid_size_y,
            img_mc_grid_size_x=img_mc_grid_size_x,
            img_mc_grid_size_y=img_mc_grid_size_y,
            normalization_factor=normalization_factor,
        )

    def get_action_space(self):
        """
        input:

        output:
            d_action_space: dictionary composition space
                the dictionary keys have to match the parameter,
                custom variable, or custom vector label.
                the value has to be defined as gymnasium space object.
                + https://gymnasium.farama.org/main/api/spaces/
        run:
            internal function, user defined.

        description:
            dictionary structure built out of gymnasium.spaces elements.
            this struct has to specify type and range for each
            action parameter, action custom variable, and action custom vector.
        """

        # model dependent action_space processing logic goes here!
        d_action_space = spaces.Dict(
            {
                "drug_1": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        # output
        return d_action_space

    def get_observation_space(self):
        """
        input:

        output:
            o_observation_space structure.
                the struct have to be built out of gymnasium.spaces elements.
                there are no other limits.
                + https://gymnasium.farama.org/main/api/spaces/

        run:
            internal function, user defined.

        description:
            data structure built out of gymnasium.spaces elements.
            this struct has to specify type and range
            for each observed variable.
        """
        # model dependent observation_space processing logic goes here!
        if self.kwargs["observation_mode"] == "scalars":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.cell_type_count,),
                dtype=np.float32,
            )

        elif self.kwargs["observation_mode"] == "img_rgb":
            # Define the Box space for the rgb alpha image
            o_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.kwargs["img_rgb_grid_size_y"],
                    self.kwargs["img_rgb_grid_size_x"],
                ),
                dtype=np.uint8,
            )

        elif self.kwargs["observation_mode"] == "img_mc":
            # Define the Box space for the multichannel image
            self.ratio_img_mc_size_y = self.height / self.kwargs["img_mc_grid_size_y"]
            self.ratio_img_mc_size_x = self.width / self.kwargs["img_mc_grid_size_x"]
            o_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.cell_type_count,
                    self.kwargs["img_mc_grid_size_x"],
                    self.kwargs["img_mc_grid_size_y"],
                ),
                dtype=np.uint8,
            )

        else:
            raise ValueError(
                f"unknown observation type: {self.kwargs['observation_mode']}"
            )

        # output
        return o_observation_space

    def get_observation(self):
        """
        input:

        output:
            o_observation: object compatible with the defined
                observation space struct.

        run:
            internal function, user defined.

        description:
            data for the observation object for example be retrieved by:
            + physicell.get_parameter("my_parameter")
            + physicell.get_variable("my_variable")
            + physicell.get_vector("my_vector")
            however, there are no limits.
        """
        # model dependent observation processing logic goes here!

        # get substrate data frame
        self.df_subs = None
        for s_subs in self.substrate_unique:
            df_subs = pd.DataFrame(
                physicell.get_microenv(s_subs), columns=["x", "y", "z", s_subs]
            )
            if self.df_subs is None:
                self.df_subs = df_subs
            else:
                self.df_subs = pd.merge(self.df_subs, df_subs, on=["x", "y", "z"])

        # get cell data frame
        self.df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "type"]
        )
        df_alive = self.df_cell[self.df_cell["dead"] < 0.1]

        # update tumor cell count
        self.c_prev = self.c_t
        self.c_t = df_alive.loc[(df_alive.type == "tumor"), :].shape[0]
        if self.c_prev is None:
            self.c_prev = self.c_t

        # update cell_1 cell count
        self.nb_cell_1 = df_alive.loc[(df_alive.type == "cell_1"), :].shape[0]

        # update cell_2 cell count
        self.nb_cell_2 = df_alive.loc[(df_alive.type == "cell_2"), :].shape[0]

        # observe the environemnt
        if self.kwargs["observation_mode"] == "scalars":
            a_norm_cell_count = np.zeros((self.cell_type_count,), dtype=np.float32)
            for s_cell_type, i_id in self.cell_type_to_id.items():
                a_norm_cell_count[i_id] = (
                    df_alive.loc[
                        (df_alive.type == s_cell_type),
                        :,
                    ].shape[0]
                    / self.kwargs["normalization_factor"]
                    - 1
                )
            o_observation = a_norm_cell_count

        elif self.kwargs["observation_mode"] == "img_rgb":
            a_img = self.render()
            a_img = ski.color.rgb2gray(ski.color.rgba2rgb(a_img))
            a_img = ski.transform.resize(  # ski.transform.rescale
                a_img,
                output_shape=(
                    self.kwargs["img_rgb_grid_size_y"],
                    self.kwargs["img_rgb_grid_size_x"],
                ),
                anti_aliasing=True,
            )
            o_observation = a_img

        elif self.kwargs["observation_mode"] == "img_mc":
            # get cell_type indices
            cell_type_indices = df_alive["type"].map(self.cell_type_to_id).to_numpy()

            # discretize
            x_bin = (
                (df_alive["x"] - self.x_min)
                / (self.x_max - self.x_min)
                * (self.kwargs["img_mc_grid_size_x"] - 1)
            ).astype(int)
            y_bin = (
                (df_alive["y"] - self.y_min)
                / (self.y_max - self.y_min)
                * (self.kwargs["img_mc_grid_size_y"] - 1)
            ).astype(int)

            # clip in case of rounding issues
            x_bin = np.clip(x_bin, 0, self.kwargs["img_mc_grid_size_x"] - 1)
            y_bin = np.clip(y_bin, 0, self.kwargs["img_mc_grid_size_y"] - 1)

            # get numpy array
            image = np.zeros(
                shape=(
                    self.cell_type_count,
                    self.kwargs["img_mc_grid_size_x"],
                    self.kwargs["img_mc_grid_size_y"],
                ),
                dtype=np.float32,
            )
            np.add.at(
                image,
                (cell_type_indices, x_bin, y_bin),
                1 / (self.ratio_img_mc_size_x * self.ratio_img_mc_size_y),
            )

            # output
            o_observation = (image * 255).astype(np.uint8)

        else:
            raise ValueError(
                f"unknown observation type: {self.kwargs['observation_mode']}"
            )

        # output
        return o_observation

    def get_info(self):
        """
        input:

        output:
            info: dictionary

        run:
            internal function, user defined.

        description:
            function to provide additional information important for
            controlling the action of the policy. for example,
            if we do reinforcement learning on a jump and run game,
            the number of hearts (lives left) from our character.
        """
        # model dependent info processing logic goes here!
        info = {
            "df_cell": self.df_cell,
            "df_subs": self.df_subs,
            "number_tumor": self.c_t,
            "number_cell_1": self.nb_cell_1,
            "number_cell_2": self.nb_cell_2,
        }

        # output
        return info

    def get_terminated(self):
        """
        input:

        output:
            b_terminated: bool

        run:
            internal function, user defined.

        description:
            function to determine if the episode is terminated.
            for example, if we do reinforcement learning on a
            jump and run game, if our character died.
            please notice, that this ending is different form
            truncated (the episode reached the max time limit).
        """
        # model dependent terminated processing logic goes here!
        return True if self.c_t == 0 else False

    def get_reset_values(self):
        """
        input:

        output:

        run:
            internal function, user defined.

        description:
            function to reset model specific self.variables. e.g.:
            self.my_variable = None
        """
        self.c_t = None
        self.c_prev = None

    def get_reward(self):
        """
        input:

        output:
            r_reward: float between or equal to 0.0 and 1.0.
                there are no other limits to the algorithm implementation enforced.
                however, the algorithm is usually based on data retrieved
                by the get_observation function (o_observation, info),
                and possibly by the render function (a_img).

        run:
            internal function, user defined.

        description:
            cost function.
        """
        if self.c_prev > 0:
            r_reward_tumor = (self.c_prev - self.c_t) / (
                self.c_prev
                * np.e
                ** (
                    physicell.get_parameter("growth_rate")
                    * physicell.get_parameter("dt_gym")
                )
                - self.c_prev
            )
            r_reward_tumor = np.clip(r_reward_tumor, -1, 1)
        else:
            r_reward_tumor = None
        return r_reward_tumor

    def get_img(self):
        """
        input:

        output:
            self.fig.savefig
                instance attached matplotlib figure.

        run:
            internal function, user defined.

        description:
            template code to generate a matplotlib figure from the data.
            for example from:
            + physicell.get_microenv("my_substrate")
            + physicell.get_cell()
            + physicell.get_variable("my_variable")
            however, there are no limits.
        """
        # model dependent img processing logic goes here!
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.axis("equal")
        ax.axis("off")

        ##################
        # substrate data #
        ##################

        # debris
        df_conc = self.df_subs.loc[self.df_subs.z == 0.0, ["x", "y", "z", "debris"]]
        df_mesh = df_conc.pivot(index="y", columns="x", values="debris")
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Reds",
            alpha=1 / 3,
        )

        # pro-inflammatory factor
        df_conc = self.df_subs.loc[
            self.df_subs.z == 0.0, ["x", "y", "z", "pro-inflammatory factor"]
        ]
        df_mesh = df_conc.pivot(
            index="y", columns="x", values="pro-inflammatory factor"
        )
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Greens",
            alpha=1 / 3,
        )

        # anti-inflammatory factor
        df_conc = self.df_subs.loc[
            self.df_subs.z == 0.0, ["x", "y", "z", "anti-inflammatory factor"]
        ]
        df_mesh = df_conc.pivot(
            index="y", columns="x", values="anti-inflammatory factor"
        )
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            alpha=1 / 3,
        )

        ######################
        # substrate colorbar #
        ######################

        # self.fig.colorbar(
        #    mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="Reds"),
        #    label="my_substrate",
        #    ax=ax,
        # )

        #############
        # cell data #
        #############

        df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "cell_type"]
        )
        df_cell = df_cell.loc[(df_cell.dead < 0.1), :]
        df_cell["color"] = None
        for s_cell_type, s_color in self.cell_type_to_color.items():
            df_cell.loc[(df_cell.cell_type == s_cell_type), "color"] = s_color
        # df_variable = pd.DataFrame(physicell.get_variable("my_variable"), columns=["my_variable"])
        # df_cell = pd.merge(df_cell, df_variable, left_index=True, right_index=True, how="left")
        df_cell = df_cell.loc[df_cell.z == 0.0, :]
        df_cell.plot(
            kind="scatter",
            x="x",
            y="y",
            c="color",
            xlim=[self.x_min, self.x_max],
            ylim=[self.y_min, self.y_max],
            #    vmin=0.0, vmax=1.0, cmap="viridis",
            #    grid=True,
            #    title=f"dt_gym env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} [cell]",
            ax=ax,
        )

        ################
        # save to file #
        ################

        # plt.tight_layout()
        # s_path = self.x_root.xpath("//save/folder")[0].text + "/render_mode_human/"
        # os.makedirs(s_path, exist_ok=True)
        # self.fig.savefig(f"{s_path}timeseries_step{str(self.step_env).zfill(3)}.jpeg", facecolor="white")
