import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os
import pandas as pd
import shutil
from init_conds import generate_initial_condition
from pathlib import Path


# ============================================================
# Wrapper: PhysiCellModelWrapper
# ============================================================
class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = ["drug_1"],
        w_cell=0.7,
        w_increase=0.2,
        w_amount=0.1,
        frequence_episode_test=3,
    ):
        """
        Wraps a PhysiCell environment to use a flat continuous Box action space.
        Reward = weighted sum between drug penalty and cancer cell signal.
        """
        super().__init__(env)

        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be str, got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name
        low = np.array(
            [
                env.action_space[variable_name].low[0]
                for variable_name in list_variable_name
            ]
        )
        high = np.array(
            [
                env.action_space[variable_name].high[0]
                for variable_name in list_variable_name
            ]
        )
        dtype = env.action_space[list_variable_name[0]].dtype

        self._action_space = Box(low=low, high=high, dtype=dtype)
        logits = np.array([w_cell, w_amount, w_increase])
        if np.sum(logits) == 1:
            self.w_cell, self.w_amount, self.w_increase = w_cell, w_amount, w_increase
        else:
            weights = np.exp(logits)
            weights /= np.sum(weights)
            self.w_cell, self.w_amount, self.w_increase = weights

        self.cell_positions_folder = (
            self.env.get_wrapper_attr("x_root")
            .xpath("//initial_conditions/cell_positions/folder")[0]
            .text
        )
        self.cell_name_file = (
            self.env.get_wrapper_attr("x_root")
            .xpath("//initial_conditions/cell_positions/filename")[0]
            .text
        )
        self.csv_path_init = os.path.join(
            self.cell_positions_folder, self.cell_name_file
        )
        self.generation_cfg = None
        self.no_generation_cfg = None
        self.generate_physicell_data = False
        self.mode = "train"  # "train" | "test"
        self.type_mode = None
        self.dataset_name = "default"
        self.base_output_dir = (
            self.env.get_wrapper_attr("x_root").xpath("//save/folder")[0].text
        )

        os.makedirs(self.base_output_dir, exist_ok=True)
        self.list_data = []
        self.seed = int(
            self.env.get_wrapper_attr("x_root").xpath("//random_seed")[0].text
        )
        self.settingxml = self.env.get_wrapper_attr("settingxml")
        self.dt_gym = float(
            self.env.get_wrapper_attr("x_root")
            .xpath("//user_parameters/dt_gym")[0]
            .text
        )
        self.frequence_episode_test = frequence_episode_test

    def change_xml(self, keys: list[str], elements: list):
        for key, element in zip(keys, elements):
            self.env.get_wrapper_attr("x_root").xpath(key)[0].text = element
        self.env.get_wrapper_attr("x_tree").write(self.settingxml, pretty_print=True)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_mode(self):
        return self.env.unwrapped.observation_mode

    def _episode_output_dir(self, run_idx: int):
        return os.path.join(
            self.base_output_dir,
            self.mode,
            "episodes",
            f"run_{str(run_idx).zfill(6)}",
        )

    def save_data(self):
        run_idx = self.env.unwrapped.episode
        if run_idx == -1:
            return
        out_dir = self._episode_output_dir(run_idx)

        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.list_data)
        if "mean_drugs" in df.columns:
            df["cumulative_mean_drugs"] = df["mean_drugs"].cumsum()
        if "reward" in df.columns:
            df["cumulative_reward"] = df["reward"].cumsum()
        df.to_csv(os.path.join(out_dir, "data.csv"), index=False)
        dst_path = os.path.join(
            out_dir,
            os.path.basename(self.csv_path_init),
        )
        shutil.copy(self.csv_path_init, dst_path)
        self.list_data = []
        self.change_xml(
            keys=["//save/folder", "//save/SVG/enable", "//save/SVG/interval"],
            elements=[
                out_dir,
                "true" if self.generate_physicell_data else "false",
                str(self.dt_gym)
                if self.generate_physicell_data
                else str(self.dt_gym * 4),
            ],
        )

    def step(self, action: np.ndarray):
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)
        drug_prev = self.info["prev_mean_drugs"]
        drug_t = np.mean(action)
        info["action"] = d_action
        drug_increase = max(0.0, drug_t - drug_prev)
        self.info["prev_mean_drugs"] = drug_t
        info["type_mode"] = self.type_mode
        info["step_episode"] = self.env.unwrapped.step_episode
        info["train_test"] = str(self.mode)

        reward = (
            self.w_cell * r_cancer_cells
            - self.w_amount * drug_t
            - self.w_increase * drug_increase
        )
        reward -= 100 if terminated and info["number_tumor"] > 100 else 0

        data = {
            "step": self.env.unwrapped.step_episode,
            "reward": reward,
            "mean_drugs": drug_t,
            "number_tumor": info["number_tumor"],
            "train_test": info["train_test"],
        }

        self.list_data.append(data)

        return obs, reward, terminated, truncated, info

    def initial_condition_generation(self, generation_cfg=None):

        # --------------------------------------------------
        # 1. Initialize base generation config ONCE
        # --------------------------------------------------
        if self.generation_cfg is None:
            if generation_cfg is None:
                raise ValueError("generation_cfg must be provided at least once")

            self.generation_cfg = generation_cfg.copy()

            # ---- env-derived spatial bounds ----
            self.generation_cfg["x_min"] = self.env.unwrapped.x_min * 0.9
            self.generation_cfg["y_min"] = self.env.unwrapped.y_min * 0.9
            self.generation_cfg["x_max"] = self.env.unwrapped.x_max * 0.9
            self.generation_cfg["y_max"] = self.env.unwrapped.y_max * 0.9
            self.mode_train = self.generation_cfg["mode_train"]
            self.mode_test = self.generation_cfg["mode_test"]
            del self.generation_cfg["mode_train"]
            del self.generation_cfg["mode_test"]
            # ---- default seed ----
            self.generation_cfg.setdefault("seed", self.seed)

            # ---- dataset name ----
            self.dataset_name = self.generation_cfg.get("dataset", "generated")

        self.generation_cfg["mode"] = (
            self.mode_train if self.mode == "train" else self.mode_test
        )

        # --------------------------------------------------
        # 2. Dataset folder (stable)
        # --------------------------------------------------
        ic_dir = os.path.join(
            self.base_output_dir,
            self.mode,
            "initial_conditions",
            self.dataset_name,
        )
        os.makedirs(ic_dir, exist_ok=True)

        # --------------------------------------------------
        # 3. Episode-specific config
        # --------------------------------------------------
        episode = self.env.unwrapped.episode
        csv_path = os.path.join(ic_dir, f"ic_{str(episode).zfill(6)}.csv")

        gen_cfg = self.generation_cfg.copy()
        gen_cfg["seed"] = self.generation_cfg["seed"] + episode
        gen_cfg["csv_path"] = csv_path

        # --------------------------------------------------
        # 4. Generate + activate
        # --------------------------------------------------
        gen_cfg["seed"] += episode
        _, self.type_mode = generate_initial_condition(**gen_cfg)
        self.update_cell_path_cell_folder(csv_path)

    def update_cell_path_cell_folder(self, path_cells_csv: str):
        p = Path(path_cells_csv)
        cell_positions_folder = str(p.parent)
        cell_name_file = p.name
        self.change_xml(
            keys=[
                "//initial_conditions/cell_positions/folder",
                "//initial_conditions/cell_positions/filename",
            ],
            elements=[
                cell_positions_folder,
                cell_name_file,
            ],
        )
        self.csv_path_init = path_cells_csv
        self.cell_name_file = cell_name_file
        self.cell_positions_folder = cell_positions_folder

    def initial_condition(self, no_generation_cfg=None):
        self.dataset_name = no_generation_cfg.get("dataset", "replay")

        if not hasattr(self, "list_csv"):
            self.list_csv = no_generation_cfg["list_csv"]
            self.current_csv_idx = 0

        csv_path = self.list_csv[self.current_csv_idx]
        self.current_csv_idx += 1

        self.update_cell_path_cell_folder(csv_path)

    def reset(
        self,
        seed=None,
        options=None,
        generation_cfg=None,
        no_generation_cfg=None,
        **kwargs,
    ):
        self.mode = (
            "test"
            if (self.env.unwrapped.episode + 1) % self.frequence_episode_test == 0
            else "train"
        )
        self.generate_physicell_data = True if self.mode == "test" else False
        if options is None:
            options = {}

        if seed is not None:
            self.seed = seed

        if generation_cfg is not None or self.generation_cfg is not None:
            self.initial_condition_generation(generation_cfg=generation_cfg)

        if no_generation_cfg is not None or self.no_generation_cfg is not None:
            self.initial_condition(no_generation_cfg=no_generation_cfg)

        self.save_data()

        self.info = {"prev_mean_drugs": 0, "train_test": ""}

        # ---- IMPORTANT: forward seed, do not invent one ----
        return self.env.reset(seed=seed, options=options)
