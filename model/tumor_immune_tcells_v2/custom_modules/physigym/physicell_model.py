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
from scipy.special import expit
from scipy.spatial import cKDTree


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
        action_mode="full",
        img_rgb_grid_size_y=64,  # pixel
        img_rgb_grid_size_x=64,  # pixel
        img_mc_grid_size_x=64,  # pixel
        img_mc_grid_size_y=64,  # pixel
        normalization_factor=512,
        k=1,
        grid_n=8,
    ):
        self.observation_mode = observation_mode
        self.grid_n = grid_n
        if "img" in observation_mode:
            self.observation_mode = observation_mode + str(
                f"_{img_mc_grid_size_x}_{img_mc_grid_size_y}"
            )
        self.k = k
        self.action_mode = action_mode

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
        if self.action_mode == "full":
            d_action_space = spaces.Dict(
                {
                    "drug_1_dose": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            )
        else:
            d_action_space = spaces.Dict(
                {
                    "drug_1_dose": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "drug_1_x": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "drug_1_y": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "drug_1_radius": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            )

        # output
        return d_action_space

    def get_dose_spent(self):
        return physicell.get_parameter("drug_1_amount_used") / (self.total_volume)

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

        elif self.observation_mode == "scalars_macrophages":
            # same as scalars_cells but macrophage count split into M1 + M2
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.cell_type_count + 1,),
                dtype=np.float32,
            )

        elif self.observation_mode == "spatial_scalars_cells":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.cell_type_count + self.cell_type_count * 6 * self.k,),
                dtype=np.float32,
            )

        elif self.observation_mode == "spatial_scalars_cells_substrates":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(
                    self.cell_type_count
                    + self.substrate_count
                    + self.cell_type_count * 6 * self.k,
                ),
                dtype=np.float32,
            )

        elif self.observation_mode == "spatial_scalars_cells_m1m2":
            # like spatial_scalars_cells, but macrophage split into M1 + M2
            # for both the scalar counts (+1) and the k-means spatial features.
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(
                    (self.cell_type_count + 1)
                    + (self.cell_type_count + 1) * 6 * self.k,
                ),
                dtype=np.float32,
            )

        elif self.observation_mode == "spatial_scalars_cells_substrates_m1m2":
            # like spatial_scalars_cells_substrates, but macrophage split M1 + M2
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(
                    (self.cell_type_count + 1)
                    + self.substrate_count
                    + (self.cell_type_count + 1) * 6 * self.k,
                ),
                dtype=np.float32,
            )

        elif (
            self.observation_mode
            == "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2"
        ):
            # like spatial_scalars_cells_spatial_no_scalars_substrates,
            # but macrophage split into M1 + M2 for cell scalars + spatial features
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(
                    (self.cell_type_count + 1)
                    + self.substrate_count * 6 * self.k
                    + (self.cell_type_count + 1) * 6 * self.k,
                ),
                dtype=np.float32,
            )

        elif self.observation_mode == "spatial_scalars_cells_spatial_substrates":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(
                    self.cell_type_count
                    + self.substrate_count
                    + self.substrate_count * 6 * self.k
                    + self.cell_type_count * 6 * self.k,
                ),
                dtype=np.float32,
            )

        elif (
            self.observation_mode
            == "spatial_scalars_cells_spatial_no_scalars_substrates"
        ):
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(
                    self.cell_type_count
                    + self.substrate_count * 6 * self.k
                    + self.cell_type_count * 6 * self.k,
                ),
                dtype=np.float32,
            )
        elif self.observation_mode == f"kmeans_spatial_scalars_cells_substrates":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(
                    self.substrate_count * 6 * self.k
                    + self.cell_type_count * 6 * self.k,
                ),
                dtype=np.float32,
            )

        elif self.observation_mode == "occupancy_grid":
            # grid_n x grid_n spatial bins per cell type + per substrate
            # cells: fraction of total population per bin (co-localization aware)
            # substrates: mean concentration per bin
            n = self.grid_n
            o_observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.cell_type_count * n * n + self.substrate_count * n * n,),
                dtype=np.float32,
            )

        elif self.observation_mode == "relational":
            # per-type:   5 (cx, cy, std_x, std_y, count_fraction)
            # per-pair:   4 (distance, sin_angle, cos_angle, quadrant_overlap)
            # per-subs:   4 (conc@tumor, grad_sin, grad_cos, mean_in_spread)
            # per-(type,subs): 1 (mean conc at cell positions)
            n_types = self.cell_type_count
            n_subs = self.substrate_count
            n_pairs = n_types * (n_types - 1) // 2
            dim = n_types * 5 + n_pairs * 4 + n_subs * 4 + n_types * n_subs * 1
            o_observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(dim,),
                dtype=np.float32,
            )

        elif self.observation_mode == "cross_nn_relational":
            # cross_nn block:  n_types * (n_types - 1) * 2  (mean + std nn dist per ordered pair)
            # relational block: same as "relational"
            n_types = self.cell_type_count
            n_subs = self.substrate_count
            n_pairs = n_types * (n_types - 1) // 2
            dim_relational = (
                n_types * 5 + n_pairs * 4 + n_subs * 4 + n_types * n_subs * 1
            )
            # ordered pairs A→B and B→A (mean_nn_dist, std_nn_dist) each
            dim_cross_nn = n_types * (n_types - 1) * 2
            o_observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(dim_relational + dim_cross_nn,),
                dtype=np.float32,
            )

        elif observation_mode in [
            f"img_mc_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_m1m2_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_substrates_m1m2_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
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
            elif (
                observation_mode
                == f"img_mc_cells_m1m2_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
            ):
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cell_type_count + 2,
                        self.kwargs["img_mc_grid_size_x"],
                        self.kwargs["img_mc_grid_size_y"],
                    ),
                    dtype=np.uint8,
                )
            elif (
                observation_mode
                == f"img_mc_cells_substrates_m1m2_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
            ):
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cell_type_count + self.substrate_count + 2,
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

    def get_spatial_substrate_features(self):
        """
        Per substrate: mean, std, min, max, and centroid (x_mean, y_mean)
        of the concentration field.
        Shape: (substrate_count * 6,)
        """
        n_subs = self.substrate_count
        features = np.zeros((n_subs * 6,), dtype=np.float32)

        x_range = self.x_max - self.x_min + 1e-8
        y_range = self.y_max - self.y_min + 1e-8

        for i, s_subs in enumerate(self.substrate_unique):
            microenv = np.asarray(physicell.get_microenv(s_subs))
            # columns: x, y, z, concentration
            x = (microenv[:, 0] - self.x_min) / x_range
            y = (microenv[:, 1] - self.y_min) / y_range
            conc = microenv[:, -1]

            total = conc.sum()

            if total < 1e-8:
                # substrate absent — all zeros, agent learns this pattern
                continue

            base = i * 6
            features[base + 0] = conc.mean()
            features[base + 1] = conc.std()
            features[base + 2] = conc.min()
            features[base + 3] = conc.max()
            # concentration-weighted centroid
            features[base + 4] = (conc * x).sum() / total  # x_centroid
            features[base + 5] = (conc * y).sum() / total  # y_centroid

        return features

    def get_cells_scalars(self):
        n_types = self.cell_type_count
        a_norm_cell_count = np.zeros((n_types,), dtype=np.float32)
        norm_factor = self.kwargs["normalization_factor"]

        for s_cell_type, i_id in self.cell_type_to_id.items():
            a_norm_cell_count[i_id] = (
                self.df_alive.loc[self.df_alive.type == s_cell_type].shape[0]
                / norm_factor
                - 1
            )

        return a_norm_cell_count

    def get_macrophage_polarization_scalars(self):
        """
        Splits macrophages into M1 (anti-tumoral dominant) and M2 (pro-tumoral
        dominant) by comparing substrate concentrations at each macrophage's
        nearest microenvironment voxel.

        Classification: pro_tumoral_factor > anti_tumoral_factor → M2, else M1.

        Returns a float32 array of shape (cell_type_count + 1,):
          - all non-macrophage types keep their normalised count (same as
            get_cells_scalars)
          - the macrophage slot is replaced by two slots: [n_M1, n_M2]
            both normalised by normalization_factor and shifted by -1
        """
        norm_factor = self.kwargs["normalization_factor"]

        # ── fetch substrate voxel positions once ─────────────────
        pro = np.asarray(
            physicell.get_microenv("pro_tumoral_factor")
        )  # (N,4): x,y,z,conc
        anti = np.asarray(physicell.get_microenv("anti_tumoral_factor"))  # (N,4)

        voxel_xy = pro[:, :2]  # (N, 2)  x,y positions
        pro_conc = pro[:, -1]  # (N,)
        anti_conc = anti[:, -1]  # (N,)

        tree = cKDTree(voxel_xy)

        # ── classify each alive macrophage ────────────────────────
        df_mac = self.df_alive[self.df_alive["type"] == "macrophage"]
        n_M1, n_M2 = 0, 0

        if len(df_mac) > 0:
            mac_xy = df_mac[["x", "y"]].to_numpy()
            _, nearest = tree.query(mac_xy)  # nearest voxel index per cell
            is_M2 = pro_conc[nearest] > anti_conc[nearest]
            n_M2 = int(is_M2.sum())
            n_M1 = len(df_mac) - n_M2

        # ── build output: non-mac counts + M1 + M2 ───────────────
        features = []
        for s_cell_type, i_id in sorted(
            self.cell_type_to_id.items(), key=lambda kv: kv[1]
        ):
            if s_cell_type == "macrophage":
                features.append(n_M1 / norm_factor - 1.0)
                features.append(n_M2 / norm_factor - 1.0)
            else:
                n = self.df_alive[self.df_alive["type"] == s_cell_type].shape[0]
                features.append(n / norm_factor - 1.0)

        return np.array(features, dtype=np.float32)

    def get_heuristic_action(self, radius_micron=10.0, dose=0.5):
        """
        Macrophage-aware rule-based baseline action (Stage 4).

        Mechanism-driven: cell_rules.csv makes drug_1 re-polarise macrophages
        (decreases pro_tumoral / increases anti_tumoral secretion), so the drug's
        leverage point is the M2 (pro-tumoral) macrophages that are feeding the
        tumour, not the tumour cells themselves. This policy therefore aims a
        fixed dose at the centroid of the M2 macrophages lying within
        `radius_micron` of any tumour cell, sizing the injection radius to that
        cluster's spread. If no such macrophage exists this step it injects
        nothing (dose = 0), so it neither wastes drug nor pays the dose penalty.

        Returns the NORMALISED [dose, x, y, radius] action in the same 0..1
        convention the wrapper expects (it denormalises x/y by x_min+ x*width and
        maps radius into [0.05, 0.20]*max_radius). This is a drop-in replacement
        for action_space.sample() and needs no learning.
        """
        no_op = np.zeros(4, dtype=np.float32)  # dose=0; x/y/radius irrelevant when dose=0

        df = self.df_alive
        df_mac = df[df["type"] == "macrophage"]
        df_tum = df[df["type"] == "tumor"]
        if len(df_mac) == 0 or len(df_tum) == 0:
            return no_op

        mac_xy = df_mac[["x", "y"]].to_numpy()
        tum_xy = df_tum[["x", "y"]].to_numpy()

        # ── classify macrophages M1/M2 (same rule as the env's scalars) ──────
        pro = np.asarray(physicell.get_microenv("pro_tumoral_factor"))  # (N,4)
        anti = np.asarray(physicell.get_microenv("anti_tumoral_factor"))
        vtree = cKDTree(pro[:, :2])
        _, nearest = vtree.query(mac_xy)
        is_M2 = pro[:, -1][nearest] > anti[:, -1][nearest]
        if not is_M2.any():
            return no_op
        m2_xy = mac_xy[is_M2]

        # ── keep only M2 macrophages within radius_micron of any tumour cell ─
        ttree = cKDTree(tum_xy)
        d_near, _ = ttree.query(m2_xy)
        adj = m2_xy[d_near <= radius_micron]
        if len(adj) == 0:
            return no_op

        # ── aim at the tumour-adjacent-M2 centroid; radius covers the cluster ─
        cx, cy = adj[:, 0].mean(), adj[:, 1].mean()
        spread = float(np.sqrt(((adj - adj.mean(axis=0)) ** 2).sum(axis=1).mean())) \
            if len(adj) > 1 else radius_micron

        # normalise to the wrapper's 0..1 convention
        x_norm = float(np.clip((cx - self.x_min) / self.width, 0.0, 1.0))
        y_norm = float(np.clip((cy - self.y_min) / self.height, 0.0, 1.0))
        max_radius = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        # invert the wrapper's radius map: physical = (0.05 + r*0.15)*max_radius
        r_phys = spread + radius_micron  # a little padding beyond the spread
        r_norm = float(np.clip((r_phys / max_radius - 0.05) / 0.15, 0.0, 1.0))

        return np.array([dose, x_norm, y_norm, r_norm], dtype=np.float32)

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
            / (self.width)
            * (self.kwargs["img_mc_grid_size_x"] - 1)
        ).astype(int)
        y_bin = (
            (df["y"] - self.y_min)
            / (self.height)
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
                / (self.width)
                * (self.kwargs["img_mc_grid_size_x"] - 1)
            )
            .astype(int)
            .clip(0, self.kwargs["img_mc_grid_size_x"] - 1)
        )
        self.df_subs["y_bin"] = (
            (
                (self.df_subs["y"] - self.y_min)
                / (self.height)
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

        return ski.util.img_as_ubyte(np.clip(image, 0, 1))

    def get_matrix_macrophage_polarization(self):
        """
        Generates M1/M2 macrophage polarization images.
        Classifies macrophages as M1 or M2 based on local substrate concentrations
        (pro-tumoral vs anti-tumoral factors) at each cell's position.

        Returns a uint8 image array of shape (2, img_mc_grid_size_x, img_mc_grid_size_y):
          - Channel 0: M1 macrophages (anti-tumoral dominant)
          - Channel 1: M2 macrophages (pro-tumoral dominant)
        """
        # ── fetch substrate voxel positions and concentrations ─────
        pro = np.asarray(
            physicell.get_microenv("pro_tumoral_factor")
        )  # (N,4): x,y,z,conc
        anti = np.asarray(physicell.get_microenv("anti_tumoral_factor"))  # (N,4)

        voxel_xy = pro[:, :2]  # (N, 2)  x,y positions
        pro_conc = pro[:, -1]  # (N,)
        anti_conc = anti[:, -1]  # (N,)

        tree = cKDTree(voxel_xy)

        # ── initialize image for M1 and M2 ────────────────────────
        image = np.zeros(
            (
                2,  # M1 and M2 channels
                self.kwargs["img_mc_grid_size_x"],
                self.kwargs["img_mc_grid_size_y"],
            ),
            dtype=np.float32,
        )

        # ── classify and bin each macrophage ──────────────────────
        df_mac = self.df_alive[self.df_alive["type"] == "macrophage"]

        if len(df_mac) > 0:
            mac_xy = df_mac[["x", "y"]].to_numpy()
            _, nearest = tree.query(mac_xy)  # nearest voxel index per cell

            # Determine M1 vs M2 for each macrophage
            is_M2 = pro_conc[nearest] > anti_conc[nearest]
            m1_mask = ~is_M2
            m2_mask = is_M2

            # Discretize macrophage positions to grid bins
            x_bin = (
                (df_mac["x"].to_numpy() - self.x_min)
                / (self.width)
                * (self.kwargs["img_mc_grid_size_x"] - 1)
            ).astype(int)
            y_bin = (
                (df_mac["y"].to_numpy() - self.y_min)
                / (self.height)
                * (self.kwargs["img_mc_grid_size_y"] - 1)
            ).astype(int)

            # Clip to grid bounds
            x_bin = np.clip(x_bin, 0, self.kwargs["img_mc_grid_size_x"] - 1)
            y_bin = np.clip(y_bin, 0, self.kwargs["img_mc_grid_size_y"] - 1)

            # Accumulate M1 and M2 counts in image channels
            # Use constant arrays with the correct shape for masked indices
            if m1_mask.sum() > 0:
                np.add.at(
                    image,
                    (
                        np.zeros(m1_mask.sum(), dtype=int),
                        x_bin[m1_mask],
                        y_bin[m1_mask],
                    ),
                    1,
                )
            if m2_mask.sum() > 0:
                np.add.at(
                    image,
                    (np.ones(m2_mask.sum(), dtype=int), x_bin[m2_mask], y_bin[m2_mask]),
                    1,
                )

        # Normalize and scale to uint8
        scaled_image = image / (self.ratio_img_mc_size_x * self.ratio_img_mc_size_y)
        clipped_image = np.clip(scaled_image, 0.0, 1.0)

        return ski.util.img_as_ubyte(clipped_image)

    def get_spatial_substrate_features(self):
        """
        Finds the top K hotspots for each substrate using concentration-weighted K-Means.
        Includes a presence flag and mass fraction for each hotspot.
        """
        n_subs = self.substrate_count
        k_clusters = self.k
        features_per_cluster = 6  # [presence, mass_fraction, cx, cy, std_x, std_y]

        features = np.zeros(
            (n_subs * k_clusters * features_per_cluster,), dtype=np.float32
        )

        x_range = self.x_max - self.x_min + 1e-8
        y_range = self.y_max - self.y_min + 1e-8

        for i, s_subs in enumerate(self.substrate_unique):
            microenv = np.asarray(physicell.get_microenv(s_subs))
            base_idx = i * k_clusters * features_per_cluster

            x = (microenv[:, 0] - self.x_min) / x_range
            y = (microenv[:, 1] - self.y_min) / y_range
            conc = microenv[:, -1]

            total_mass = conc.sum()

            # Filter out background noise to speed up K-Means
            # Only cluster points that have at least 1% of the max concentration
            max_conc = conc.max()
            threshold = 0.01 * max_conc if max_conc > 1e-8 else 1.0
            mask = conc > threshold

            if total_mass < 1e-8 or not mask.any():
                # Substrate absent — array stays 0.0
                continue

            valid_x = x[mask]
            valid_y = y[mask]
            valid_conc = conc[mask]
            coords = np.column_stack((valid_x, valid_y))

            actual_k = min(k_clusters, len(valid_x))

            # The Edge: Weight the K-Means points by their chemical concentration
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=1)
            labels = kmeans.fit_predict(coords, sample_weight=valid_conc)
            centers = kmeans.cluster_centers_

            cluster_stats = []
            for c in range(actual_k):
                c_mask = labels == c
                if not c_mask.any():
                    continue

                c_conc = valid_conc[c_mask]
                c_x = valid_x[c_mask]
                c_y = valid_y[c_mask]

                presence = 1.0
                mass_fraction = c_conc.sum() / total_mass
                cx, cy = centers[c]

                # Weighted standard deviation to measure the spread of the plume
                if len(c_conc) > 1 and c_conc.sum() > 0:
                    c_std_x = np.sqrt(np.average((c_x - cx) ** 2, weights=c_conc))
                    c_std_y = np.sqrt(np.average((c_y - cy) ** 2, weights=c_conc))
                else:
                    c_std_x, c_std_y = 0.0, 0.0

                cluster_stats.append(
                    (presence, mass_fraction, cx, cy, c_std_x, c_std_y)
                )

            # Sort hotspots descending by mass fraction (largest hotspot → slot 0)
            cluster_stats.sort(key=lambda item: item[1], reverse=True)

            # Populate array
            for c, stats in enumerate(cluster_stats):
                idx = base_idx + (c * features_per_cluster)
                features[idx : idx + features_per_cluster] = stats

        return features

    def get_spatial_features(self):
        """
        Extracts spatial features using K-Means clustering.
        Sorted by global weight, with an explicit presence flag per cluster.
        Cluster mass is normalized against the total cell population.
        """
        n_types = self.cell_type_count
        k_clusters = self.k
        features_per_cluster = 6  # [presence, global_weight, cx, cy, std_x, std_y]

        features = np.zeros(
            (n_types * k_clusters * features_per_cluster,), dtype=np.float32
        )

        # The Edge: Get total alive cells across ALL types for true global normalization
        total_alive_cells = len(self.df_alive)

        for s_cell_type, i_id in self.cell_type_to_id.items():
            df_type = self.df_alive[self.df_alive["type"] == s_cell_type]
            base_idx = i_id * k_clusters * features_per_cluster

            n_cells = len(df_type)
            if n_cells == 0 or total_alive_cells == 0:
                # Type absent — all k_clusters stay 0.0 (presence = 0.0)
                continue

            x = (df_type["x"].to_numpy() - self.x_min) / self.width
            y = (df_type["y"].to_numpy() - self.y_min) / self.height
            coords = np.column_stack((x, y))

            actual_k = min(k_clusters, n_cells)
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=1)
            labels = kmeans.fit_predict(coords)
            centers = kmeans.cluster_centers_

            cluster_stats = []
            for c in range(actual_k):
                c_mask = labels == c
                c_coords = coords[c_mask]

                presence = 1.0

                # Normalizing cluster size by the TOTAL cell population
                global_weight = len(c_coords) / total_alive_cells

                cx, cy = centers[c]

                c_std_x = c_coords[:, 0].std() if len(c_coords) > 1 else 0.0
                c_std_y = c_coords[:, 1].std() if len(c_coords) > 1 else 0.0

                cluster_stats.append(
                    (presence, global_weight, cx, cy, c_std_x, c_std_y)
                )

            # Sort descending by global weight to maintain stability in the state array
            cluster_stats.sort(key=lambda item: item[1], reverse=True)

            # Populate array
            for c, stats in enumerate(cluster_stats):
                idx = base_idx + (c * features_per_cluster)
                features[idx : idx + features_per_cluster] = stats

        return features

    def get_spatial_features_m1m2(self):
        """
        Same as get_spatial_features, but the single "macrophage" cell type is
        split into two spatially independent groups, M1 (anti-tumoral dominant)
        and M2 (pro-tumoral dominant), classified per macrophage by comparing
        pro/anti tumoral substrate concentrations at its nearest voxel (same
        rule as get_macrophage_polarization_scalars). Every other cell type is
        clustered exactly as before.

        Layout: cell_type_count + 1 groups (macrophage -> M1, M2), each with
        k_clusters * 6 features ([presence, global_weight, cx, cy, std_x, std_y]).
        Group order matches cell_type_to_id, with M1 taking the macrophage slot
        and M2 appended as one extra group at the end.
        """
        n_groups = self.cell_type_count + 1  # macrophage split into M1 + M2
        k_clusters = self.k
        features_per_cluster = 6  # [presence, global_weight, cx, cy, std_x, std_y]

        features = np.zeros(
            (n_groups * k_clusters * features_per_cluster,), dtype=np.float32
        )

        total_alive_cells = len(self.df_alive)
        if total_alive_cells == 0:
            return features

        # ── classify macrophages into M1 / M2 by nearest-voxel substrate ──
        df_mac = self.df_alive[self.df_alive["type"] == "macrophage"]
        mac_is_m2 = np.zeros(len(df_mac), dtype=bool)
        if len(df_mac) > 0:
            pro = np.asarray(physicell.get_microenv("pro_tumoral_factor"))
            anti = np.asarray(physicell.get_microenv("anti_tumoral_factor"))
            tree = cKDTree(pro[:, :2])
            _, nearest = tree.query(df_mac[["x", "y"]].to_numpy())
            mac_is_m2 = pro[:, -1][nearest] > anti[:, -1][nearest]

        # ── build the list of (group_index, dataframe) to cluster ──
        # non-macrophage types keep their cell_type_to_id slot; macrophage is
        # replaced by M1 (its own slot) and M2 (appended extra slot).
        m2_group_idx = self.cell_type_count  # extra slot appended at the end
        groups = []
        for s_cell_type, i_id in self.cell_type_to_id.items():
            if s_cell_type == "macrophage":
                groups.append((i_id, df_mac[~mac_is_m2]))  # M1
                groups.append((m2_group_idx, df_mac[mac_is_m2]))  # M2
            else:
                groups.append(
                    (i_id, self.df_alive[self.df_alive["type"] == s_cell_type])
                )

        for group_idx, df_group in groups:
            base_idx = group_idx * k_clusters * features_per_cluster
            n_cells = len(df_group)
            if n_cells == 0:
                continue

            x = (df_group["x"].to_numpy() - self.x_min) / self.width
            y = (df_group["y"].to_numpy() - self.y_min) / self.height
            coords = np.column_stack((x, y))

            actual_k = min(k_clusters, n_cells)
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=1)
            labels = kmeans.fit_predict(coords)
            centers = kmeans.cluster_centers_

            cluster_stats = []
            for c in range(actual_k):
                c_mask = labels == c
                c_coords = coords[c_mask]

                presence = 1.0
                global_weight = len(c_coords) / total_alive_cells
                cx, cy = centers[c]
                c_std_x = c_coords[:, 0].std() if len(c_coords) > 1 else 0.0
                c_std_y = c_coords[:, 1].std() if len(c_coords) > 1 else 0.0

                cluster_stats.append(
                    (presence, global_weight, cx, cy, c_std_x, c_std_y)
                )

            cluster_stats.sort(key=lambda item: item[1], reverse=True)

            for c, stats in enumerate(cluster_stats):
                idx = base_idx + (c * features_per_cluster)
                features[idx : idx + features_per_cluster] = stats

        return features

    def get_relational_features(self):
        """
        48-dimensional interpretable relational state vector.

        Block 1 — per cell type (5 each):
          cx, cy           : concentration-weighted centroid, normalised to [0,1]
          std_x, std_y     : spread of cell positions, normalised by domain size
          count_fraction   : fraction of total alive population

        Block 2 — per ordered pair of cell types (4 each, C(n,2) pairs):
          distance         : Euclidean centroid distance, normalised by domain diagonal
          sin_angle        : sine of angle from type-A centroid to type-B centroid
          cos_angle        : cosine of same angle
          quadrant_overlap : fraction of cells of type-A in same quadrant as type-B centroid

        Block 3 — per substrate, drug-relative (4 each):
          conc_at_tumor    : mean substrate concentration at tumor cell positions
          grad_sin         : sin of direction of max concentration gradient
          grad_cos         : cos of same gradient direction
          mean_in_spread   : mean concentration within 1-std radius of tumor centroid

        Block 4 — per (cell type, substrate) cross (1 each):
          mean_conc_at_type: mean substrate concentration sampled at each cell's position
        """
        x_range = self.x_max - self.x_min + 1e-8
        y_range = self.y_max - self.y_min + 1e-8
        diagonal = np.sqrt(x_range**2 + y_range**2) + 1e-8
        total_alive = max(len(self.df_alive), 1)

        # ── Pre-compute per-type centroids and spreads ────────────
        type_names = list(self.cell_type_to_id.keys())  # stable order
        centroids = {}  # name → (cx_norm, cy_norm)
        spreads = {}  # name → (sx_norm, sy_norm)
        counts = {}  # name → int

        for s in type_names:
            df_t = self.df_alive[self.df_alive["type"] == s]
            n = len(df_t)
            counts[s] = n
            if n == 0:
                centroids[s] = (0.5, 0.5)  # domain centre as neutral
                spreads[s] = (0.0, 0.0)
            else:
                cx = (df_t["x"].mean() - self.x_min) / x_range
                cy = (df_t["y"].mean() - self.y_min) / y_range
                sx = df_t["x"].std(ddof=0) / x_range if n > 1 else 0.0
                sy = df_t["y"].std(ddof=0) / y_range if n > 1 else 0.0
                centroids[s] = (float(cx), float(cy))
                spreads[s] = (float(sx), float(sy))

        # ── Block 1: per-type features ────────────────────────────
        block1 = []
        for s in type_names:
            cx, cy = centroids[s]
            sx, sy = spreads[s]
            cf = counts[s] / total_alive
            block1.extend([cx, cy, sx, sy, cf])

        # ── Block 2: per-pair relational features ─────────────────
        block2 = []
        for i in range(len(type_names)):
            for j in range(i + 1, len(type_names)):
                sa, sb = type_names[i], type_names[j]
                ax, ay = centroids[sa]
                bx, by = centroids[sb]

                dx = bx - ax
                dy = by - ay
                dist = np.sqrt(dx**2 + dy**2) / np.sqrt(
                    2.0
                )  # max diagonal in [0,1]² = sqrt(2)
                angle = np.arctan2(dy, dx)
                sin_a = float(np.sin(angle))
                cos_a = float(np.cos(angle))

                # fraction of type-A cells in the same quadrant as type-B centroid
                df_a = self.df_alive[self.df_alive["type"] == sa]
                if len(df_a) == 0:
                    overlap = 0.0
                else:
                    # quadrant defined by domain centre
                    qx = bx > 0.5  # type-B is left or right half
                    qy = by > 0.5
                    xa_norm = (df_a["x"].to_numpy() - self.x_min) / x_range
                    ya_norm = (df_a["y"].to_numpy() - self.y_min) / y_range
                    in_q = ((xa_norm > 0.5) == qx) & ((ya_norm > 0.5) == qy)
                    overlap = float(in_q.mean())

                block2.extend([float(dist), sin_a, cos_a, overlap])

        # ── Pre-compute substrate microenv arrays + KDTree once ──
        sub_arrays = {}
        for s_subs in self.substrate_unique:
            me = np.asarray(physicell.get_microenv(s_subs))
            xn = (me[:, 0] - self.x_min) / x_range
            yn = (me[:, 1] - self.y_min) / y_range
            tree = cKDTree(np.column_stack((xn, yn)))
            sub_arrays[s_subs] = (
                xn,
                yn,
                me[:, -1],
                tree,
            )  # (x_norm, y_norm, conc, tree)

        # ── Block 3: per-substrate, drug-relative features ────────
        tumor_cx, tumor_cy = centroids.get("tumor", (0.5, 0.5))
        tumor_sx, tumor_sy = spreads.get("tumor", (0.1, 0.1))
        tumor_spread = max(np.sqrt(tumor_sx**2 + tumor_sy**2), 1e-3)

        block3 = []
        for s_subs in self.substrate_unique:
            xn, yn, conc, tree = sub_arrays[s_subs]

            # concentration at tumor cell positions (nearest voxel via KDTree)
            df_tumor = (
                self.df_alive[self.df_alive["type"] == "tumor"]
                if "tumor" in self.cell_type_to_id
                else pd.DataFrame()
            )
            if len(df_tumor) == 0:
                conc_at_tumor = 0.0
            else:
                txn = (df_tumor["x"].to_numpy() - self.x_min) / x_range
                tyn = (df_tumor["y"].to_numpy() - self.y_min) / y_range
                _, nearest = tree.query(np.column_stack((txn, tyn)))
                conc_at_tumor = float(conc[nearest].mean())

            # gradient direction: weighted centroid of top-25% concentration voxels
            thresh = np.percentile(conc, 75) if conc.max() > 1e-8 else 1.0
            hot = conc >= thresh
            if hot.sum() > 0 and conc.max() > 1e-8:
                gc_x = float(np.average(xn[hot], weights=conc[hot]))
                gc_y = float(np.average(yn[hot], weights=conc[hot]))
                gdx = gc_x - tumor_cx
                gdy = gc_y - tumor_cy
                gang = np.arctan2(gdy, gdx)
                grad_sin = float(np.sin(gang))
                grad_cos = float(np.cos(gang))
            else:
                grad_sin, grad_cos = 0.0, 0.0

            # mean concentration within 1-std radius of tumor centroid
            dist_to_tumor = np.sqrt((xn - tumor_cx) ** 2 + (yn - tumor_cy) ** 2)
            in_spread = dist_to_tumor <= tumor_spread
            mean_in_spread = float(conc[in_spread].mean()) if in_spread.any() else 0.0

            block3.extend(
                [
                    np.clip(conc_at_tumor, 0.0, 1.0),
                    grad_sin,
                    grad_cos,
                    np.clip(mean_in_spread, 0.0, 1.0),
                ]
            )

        # ── Block 4: mean substrate at each cell type's positions ──
        block4 = []
        for s_cell in type_names:
            df_t = self.df_alive[self.df_alive["type"] == s_cell]
            for s_subs in self.substrate_unique:
                xn, yn, conc, tree = sub_arrays[s_subs]
                if len(df_t) == 0 or conc.max() < 1e-8:
                    block4.append(0.0)
                    continue
                cxn = (df_t["x"].to_numpy() - self.x_min) / x_range
                cyn = (df_t["y"].to_numpy() - self.y_min) / y_range
                _, nearest = tree.query(np.column_stack((cxn, cyn)))
                block4.append(float(np.clip(conc[nearest].mean(), 0.0, 1.0)))

        return np.array(block1 + block2 + block3 + block4, dtype=np.float32)

    def get_cross_nn_features(self):
        """
        For every ordered pair (A, B) of cell types, compute the mean and std
        of the distance from each A-cell to its nearest B-cell neighbour.

        This is the vector equivalent of what a CNN learns implicitly from
        multi-channel images: how close are populations, and how variable is
        that proximity (uniform infiltration vs. sparse contact)?

        Shape: n_types * (n_types - 1) * 2
          — 2 values (mean_nn_dist, std_nn_dist) per ordered pair
          — all distances normalised by domain diagonal → [0, 1]
          — absent-type slots stay 0.0
        """
        x_range = self.x_max - self.x_min + 1e-8
        y_range = self.y_max - self.y_min + 1e-8
        diagonal = np.sqrt(x_range**2 + y_range**2) + 1e-8

        type_names = list(self.cell_type_to_id.keys())
        n_types = len(type_names)

        # Pre-build normalised coords + KDTree per type
        coords = {}
        trees = {}
        for s in type_names:
            df_t = self.df_alive[self.df_alive["type"] == s]
            if len(df_t) == 0:
                coords[s] = None
                trees[s] = None
            else:
                xn = (df_t["x"].to_numpy() - self.x_min) / x_range
                yn = (df_t["y"].to_numpy() - self.y_min) / y_range
                coords[s] = np.column_stack((xn, yn))
                trees[s] = cKDTree(coords[s])

        features = []
        for sa in type_names:
            for sb in type_names:
                if sa == sb:
                    continue
                if coords[sa] is None or trees[sb] is None:
                    features.extend([0.0, 0.0])
                    continue
                # distance from each A-cell to nearest B-cell, in diagonal units
                dists, _ = trees[sb].query(coords[sa], k=1)
                dists_norm = dists / diagonal
                features.append(float(dists_norm.mean()))
                features.append(float(dists_norm.std()) if len(dists_norm) > 1 else 0.0)

        return np.array(features, dtype=np.float32)

    def get_occupancy_grid(self):
        """
        Encodes both cell positions and substrate concentrations into a flat
        grid_n x grid_n occupancy map per channel.

        Cell channels  : fraction of total alive population in each bin.
                         (normalised by total so values are in [0, 1])
        Substrate channels: mean concentration in each bin, clipped to [0, 1].

        Shape: (cell_type_count + substrate_count) * grid_n * grid_n
        Co-localisation is implicit: if tumour cells and T-cells share a bin,
        both channels are non-zero at the same index — the MLP sees this.
        """
        n = self.grid_n
        n_types = self.cell_type_count
        n_subs = self.substrate_count
        total_cells = max(len(self.df_alive), 1)

        cell_grid = np.zeros((n_types, n, n), dtype=np.float32)
        for s_cell_type, i_id in self.cell_type_to_id.items():
            df_type = self.df_alive[self.df_alive["type"] == s_cell_type]
            if len(df_type) == 0:
                continue
            xi = (
                ((df_type["x"].to_numpy() - self.x_min) / self.width * n)
                .astype(int)
                .clip(0, n - 1)
            )
            yi = (
                ((df_type["y"].to_numpy() - self.y_min) / self.height * n)
                .astype(int)
                .clip(0, n - 1)
            )
            np.add.at(cell_grid[i_id], (xi, yi), 1.0)
        cell_grid /= total_cells  # normalise → each value in [0, 1]

        subs_grid = np.zeros((n_subs, n, n), dtype=np.float32)
        counts = np.zeros((n, n), dtype=np.float32)
        for i, s_subs in enumerate(self.substrate_unique):
            microenv = np.asarray(physicell.get_microenv(s_subs))
            x = microenv[:, 0]
            y = microenv[:, 1]
            conc = microenv[:, -1]
            xi = ((x - self.x_min) / self.width * n).astype(int).clip(0, n - 1)
            yi = ((y - self.y_min) / self.height * n).astype(int).clip(0, n - 1)
            counts[:] = 0.0
            np.add.at(subs_grid[i], (xi, yi), conc)
            np.add.at(counts, (xi, yi), 1.0)
            mask = counts > 0
            subs_grid[i][mask] /= counts[mask]  # mean concentration per bin
        subs_grid = np.clip(subs_grid, 0.0, 1.0)

        return np.concatenate([cell_grid.ravel(), subs_grid.ravel()])

    def get_observation(self):
        """expit
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
            self.c_0 = self.c_t
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
        elif self.observation_mode == "scalars_macrophages":
            o_observation = self.get_macrophage_polarization_scalars()
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
        elif (
            self.observation_mode
            == f"img_mc_cells_m1m2_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = np.concatenate(
                [
                    self.get_matrix_cells(),
                    self.get_matrix_macrophage_polarization(),
                ]
            )
        elif (
            self.observation_mode
            == f"img_mc_cells_substrates_m1m2_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = np.concatenate(
                [
                    self.get_matrix_cells(),
                    self.get_matrix_substrates(),
                    self.get_matrix_macrophage_polarization(),
                ]
            )
        elif self.observation_mode == "spatial_scalars_cells":
            o_observation = np.concatenate(
                [
                    self.get_cells_scalars(),
                    self.get_spatial_features(),
                ]
            )

        elif self.observation_mode == "spatial_scalars_cells_substrates":
            o_observation = np.concatenate(
                [
                    self.get_cells_scalars(),
                    self.get_substrates_scalars(),
                    self.get_spatial_features(),
                ]
            )

        elif self.observation_mode == "spatial_scalars_cells_m1m2":
            o_observation = np.concatenate(
                [
                    self.get_macrophage_polarization_scalars(),
                    self.get_spatial_features_m1m2(),
                ]
            )

        elif self.observation_mode == "spatial_scalars_cells_substrates_m1m2":
            o_observation = np.concatenate(
                [
                    self.get_macrophage_polarization_scalars(),
                    self.get_substrates_scalars(),
                    self.get_spatial_features_m1m2(),
                ]
            )

        elif (
            self.observation_mode
            == "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2"
        ):
            o_observation = np.concatenate(
                [
                    self.get_macrophage_polarization_scalars(),
                    self.get_spatial_features_m1m2(),
                    self.get_spatial_substrate_features(),
                ]
            )

        elif self.observation_mode == "spatial_scalars_cells_spatial_substrates":
            o_observation = np.concatenate(
                [
                    self.get_cells_scalars(),
                    self.get_substrates_scalars(),
                    self.get_spatial_features(),
                    self.get_spatial_substrate_features(),
                ]
            )
        elif (
            self.observation_mode
            == "spatial_scalars_cells_spatial_no_scalars_substrates"
        ):
            o_observation = np.concatenate(
                [
                    self.get_cells_scalars(),
                    self.get_spatial_features(),
                    self.get_spatial_substrate_features(),
                ]
            )

        elif self.observation_mode == f"kmeans_spatial_scalars_cells_substrates":
            o_observation = np.concatenate(
                [
                    self.get_spatial_features(),
                    self.get_spatial_substrate_features(),
                ]
            )

        elif self.observation_mode == "occupancy_grid":
            o_observation = self.get_occupancy_grid()

        elif self.observation_mode == "relational":
            o_observation = self.get_relational_features()

        elif self.observation_mode == "cross_nn_relational":
            o_observation = np.concatenate(
                [
                    self.get_relational_features(),
                    self.get_cross_nn_features(),
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
        return True if self.c_t <= 3 or self.c_t > 256 else False

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
        self.c_0 = None

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
        return r_tumor

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
            (df_cell["x"] - self.x_min) / (self.width) * (canvas_width - 100)
        ).astype(int)
        y_scaled = (
            (df_cell["y"] - self.y_min) / (self.height) * (canvas_height - 20)
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
