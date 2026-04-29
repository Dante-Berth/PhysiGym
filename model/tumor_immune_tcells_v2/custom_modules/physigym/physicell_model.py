#####
# title: physigym/envs/physicell_model.py
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
import numpy as np
import os
import pandas as pd
from physigym.envs.physicell_core import CorePhysiCellEnv
import skimage as ski
from tysserand import tysserand as ty
from sklearn.cluster import KMeans
import cv2
from numpy.fft import fft2, fftshift


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
        figsize=(6, 6),  # inch
        render_mode=None,
        render_fps=10,
        verbose=True,
        # **kwargs
        observation_mode="scalars_cells_substrates",
        img_rgb_grid_size_y=64,  # pixel
        img_rgb_grid_size_x=64,  # pixel
        img_mc_grid_size_x=64,  # pixel
        img_mc_grid_size_y=64,  # pixel
        normalization_factor=512,
    ):
        self.observation_mode = observation_mode
        if "img" in observation_mode:
            self.observation_mode = observation_mode + str(
                f"_{img_mc_grid_size_x}_{img_mc_grid_size_y}"
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
            img_rgb_grid_size_x=img_rgb_grid_size_x,
            img_rgb_grid_size_y=img_rgb_grid_size_y,
            img_mc_grid_size_x=img_mc_grid_size_x,
            img_mc_grid_size_y=img_mc_grid_size_y,
            normalization_factor=normalization_factor,
        )
        self.lambda_dt = float(
            self.x_root.xpath("//user_parameters/growth_rate")[0].text
        ) * float(self.x_root.xpath("//user_parameters/dt_gym")[0].text)

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
        observation_mode = self.observation_mode
        self.kwargs["img_mc_grid_size_x"] = self.kwargs["img_mc_grid_size_x"]
        self.kwargs["img_mc_grid_size_y"] = self.kwargs["img_mc_grid_size_y"]
        self.ratio_img_mc_size_y = self.height / self.kwargs["img_mc_grid_size_y"]
        self.ratio_img_mc_size_x = self.width / self.kwargs["img_mc_grid_size_x"]
        # model dependent observation_space processing logic goes here!

        if self.observation_mode == "scalars_cells":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.cell_type_count,),
                dtype=np.float32,
            )

        elif self.observation_mode == "scalars_substrates":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.substrate_count,),
                dtype=np.float32,
            )

        elif self.observation_mode in "scalars_cells_substrates":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.cell_type_count + self.substrate_count,),
                dtype=np.float32,
            )

        elif observation_mode in [
            f"img_mc_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
        ]:
            if (
                observation_mode
                == f"img_mc_cells_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
            ):
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
            elif (
                observation_mode
                == f"img_mc_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
            ):
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.substrate_count,
                        self.kwargs["img_mc_grid_size_x"],
                        self.kwargs["img_mc_grid_size_y"],
                    ),
                    dtype=np.uint8,
                )
            else:
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cell_type_count + self.substrate_count,
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

    def get_cells_scalars(self):
        # Initialize the array for both alive and dead counts
        n_types = self.cell_type_count
        a_norm_cell_count = np.zeros((n_types,), dtype=np.float32)
        norm_factor = self.kwargs["normalization_factor"]

        for s_cell_type, i_id in self.cell_type_to_id.items():
            # Store alive counts in the first half: [0 to n_types-1]
            a_norm_cell_count[i_id] = (
                self.df_alive.loc[self.df_alive.type == s_cell_type].shape[0]
                / norm_factor
                - 1
            )

        return a_norm_cell_count

    def get_substrates_scalars(self):
        a_substrate = np.zeros(self.substrate_count, dtype=np.float32)

        for i, s_subs in enumerate(self.substrate_unique):
            microenv = np.asarray(physicell.get_microenv(s_subs))
            values = microenv[:, -1]  # substrate column
            a_substrate[i] = np.max(values)

        return a_substrate

    def get_matrix(self, df):
        cell_type_indices = df["type"].map(self.cell_type_to_id).to_numpy()

        # discretize
        x_bin = (
            (df["x"] - self.x_min)
            / (self.x_max - self.x_min)
            * (self.kwargs["img_mc_grid_size_x"] - 1)
        ).astype(int)
        y_bin = (
            (df["y"] - self.y_min)
            / (self.y_max - self.y_min)
            * (self.kwargs["img_mc_grid_size_y"] - 1)
        ).astype(int)

        # FIXED: clip to size - 1 (e.g., 63, not 64)
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
            1,
        )

        # FIXED: Prevent floats > 1.0 from crashing ski.util.img_as_ubyte
        scaled_image = image / (self.ratio_img_mc_size_x * self.ratio_img_mc_size_y)
        clipped_image = np.clip(scaled_image, 0.0, 1.0)

        return ski.util.img_as_ubyte(clipped_image)

    def get_matrix_cells(self):
        df = self.df_alive
        return self.get_matrix(df=df)

    def get_matrix_substrates(self):
        self.df_subs = None
        for s_subs in self.substrate_unique:
            df_subs = pd.DataFrame(
                physicell.get_microenv(s_subs), columns=["x", "y", "z", s_subs]
            )
            if self.df_subs is None:
                self.df_subs = df_subs
            else:
                self.df_subs = pd.merge(self.df_subs, df_subs, on=["x", "y", "z"])
        # discretize
        self.df_subs["x_bin"] = (
            (
                (self.df_subs["x"] - self.x_min)
                / (self.x_max - self.x_min)
                * (self.kwargs["img_mc_grid_size_x"] - 1)
            )
            .astype(int)
            .clip(0, self.kwargs["img_mc_grid_size_x"] - 1)
        )
        self.df_subs["y_bin"] = (
            (
                (self.df_subs["y"] - self.y_min)
                / (self.y_max - self.y_min)
                * (self.kwargs["img_mc_grid_size_y"] - 1)
            )
            .astype(int)
            .clip(0, self.kwargs["img_mc_grid_size_y"] - 1)
        )

        grouped = self.df_subs.groupby(["x_bin", "y_bin"])[self.substrate_unique].max()

        # initialize image
        image = np.zeros(
            (
                len(self.substrate_unique),
                self.kwargs["img_mc_grid_size_x"],
                self.kwargs["img_mc_grid_size_y"],
            ),
            dtype=np.float32,
        )

        # fill image
        for i, subs in enumerate(self.substrate_unique):
            for (x_bin, y_bin), value in grouped[subs].items():
                image[i, x_bin, y_bin] = value
        min_vals = image.min(axis=(1, 2), keepdims=True)
        max_vals = image.max(axis=(1, 2), keepdims=True)
        scales = np.where((max_vals - min_vals) > 0, max_vals - min_vals, 1)
        return ski.util.img_as_ubyte(((image - min_vals) / scales))

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

        # get cell data frame
        self.df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "type"]
        )
        self.df_dead = self.df_cell[self.df_cell["dead"] >= 0.1]
        self.df_alive = self.df_cell[self.df_cell["dead"] < 0.1]

        # update tumor cell count
        self.c_prev = self.c_t
        self.c_t = self.df_alive.loc[(self.df_alive.type == "tumor"), :].shape[0]
        if self.c_prev is None:
            self.c_prev = self.c_t
        self.nb_tumor = self.c_t

        # observe the environemnt
        if self.observation_mode == "scalars_cells":
            o_observation = self.get_cells_scalars()
        elif self.observation_mode == "scalars_substrates":
            o_observation = self.get_substrates_scalars()
        elif self.observation_mode == "scalars_cells_substrates":
            o_observation = np.concatenate(
                [self.get_cells_scalars(), self.get_substrates_scalars()]
            )
        elif (
            self.observation_mode
            == f"img_mc_cells_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = self.get_matrix_cells()
        elif (
            self.observation_mode
            == f"img_mc_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = self.get_matrix_substrates()
        elif (
            self.observation_mode
            == f"img_mc_cells_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = np.concatenate(
                [
                    self.get_matrix_cells(),
                    self.get_matrix_substrates(),
                ]
            )

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
            "number_tumor": self.nb_tumor,
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
        return True if (self.c_t > 1024) or (self.c_t == 0) else False

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

        expected_growth = self.c_prev * (np.exp(self.lambda_dt) - 1.0)
        expected_growth = max(expected_growth, 1e-8)

        r_tumor = (self.c_prev - self.c_t) / expected_growth
        return 1 / (1 + np.exp(-r_tumor))

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
        df_conc = pd.DataFrame(
            physicell.get_microenv("debris"), columns=["x", "y", "z", "debris"]
        )
        df_conc = df_conc.loc[df_conc.z == 0.0, :]
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

        # pro-tumoral factor
        df_conc = pd.DataFrame(
            physicell.get_microenv("pro-tumoral factor"),
            columns=["x", "y", "z", "pro-tumoral factor"],
        )
        df_conc = df_conc.loc[df_conc.z == 0.0, :]
        df_mesh = df_conc.pivot(index="y", columns="x", values="pro-tumoral factor")
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            alpha=1 / 3,
        )

        # anti-tumoral factor
        df_conc = pd.DataFrame(
            physicell.get_microenv("anti-tumoral factor"),
            columns=["x", "y", "z", "anti-tumoral factor"],
        )
        df_conc = df_conc.loc[df_conc.z == 0.0, :]
        df_mesh = df_conc.pivot(index="y", columns="x", values="anti-tumoral factor")
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Greens",
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
            #    title=f"dt_self.kwargs['img_mc_grid_size_y']m env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} [cell]",
            ax=ax,
        )

        ################
        # save to file #
        ################

        plt.tight_layout()
        s_path = self.x_root.xpath("//save/folder")[0].text + "/render_mode_human/"
        os.makedirs(s_path, exist_ok=True)
        self.fig.savefig(
            f"{s_path}timeseries_step{str(self.step_env).zfill(3)}.jpeg",
            facecolor="white",
        )

    def save_fig(self, action_value: float):
        """
        Fast rendering of cells + action bar using OpenCV (no matplotlib).
        Saves a JPEG frame ready for video creation.
        """

        # Canvas settings
        canvas_width, canvas_height = 800, 800
        canvas = (
            np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        )  # white background

        # Scale cell coordinates to canvas
        df_cell = self.df_alive.copy()
        df_cell = df_cell[df_cell.z == 0.0]  # only z=0

        x_scaled = (
            (df_cell["x"] - self.x_min)
            / (self.x_max - self.x_min)
            * (canvas_width - 100)
        ).astype(int)
        y_scaled = (
            (df_cell["y"] - self.y_min)
            / (self.y_max - self.y_min)
            * (canvas_height - 20)
        ).astype(int)

        # Draw cells
        for x, y, cell_type in zip(x_scaled, y_scaled, df_cell["type"]):
            color = self.cell_type_to_color[cell_type]
            # Convert RGB [0,1] to BGR [0,255] for OpenCV
            if isinstance(color, (tuple, list)):
                bgr = tuple(int(255 * val) for val in reversed(color))
            else:
                bgr = (0, 0, 255)  # default red
            cv2.circle(
                canvas, (x, canvas_height - 1 - y), 3, bgr, -1
            )  # invert y for OpenCV coords

        # Draw action bar
        action_space = self.get_action_space()["drug_1"]
        action_min, action_max = float(action_space.low[0]), float(action_space.high[0])
        action_scaled = int(
            ((action_value - action_min) / (action_max - action_min)) * canvas_height
        )
        action_scaled = np.clip(action_scaled, 0, canvas_height)

        bar_x_start = canvas_width - 50
        bar_width = 20
        cv2.rectangle(
            canvas,
            (bar_x_start, canvas_height - action_scaled),
            (bar_x_start + bar_width, canvas_height),
            (0, 0, 255),
            -1,
        )

        # Optional: write action value text
        font_scale = 0.5
        cv2.putText(
            canvas,
            f"{action_value:.2f}",
            (bar_x_start + bar_width + 5, canvas_height - action_scaled // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Save JPEG frame
        s_path = os.path.join(
            self.x_root.xpath("//save/folder")[0].text, "render_mode_human"
        )
        os.makedirs(s_path, exist_ok=True)
        filename = f"{s_path}/timeseries_step{str(self.step_env).zfill(3)}.jpeg"
        cv2.imwrite(filename, canvas)
