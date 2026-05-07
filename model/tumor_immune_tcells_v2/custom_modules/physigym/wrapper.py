import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os
import pandas as pd
import shutil
from init_conds_v3 import generate_initial_condition
from pathlib import Path

# ============================================================
# Wrapper: PhysiCellModelWrapper
# ============================================================


class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_1_dose",
            "drug_1_x",
            "drug_1_y",
            "drug_1_radius",
        ],
        w_cell=0.5,
        frequence_episode_test=3,
    ):
        """
        Wraps a PhysiCell environment to use a flat continuous Box action space.
        Ensures XML and CSV paths are updated before the simulation starts.
        """
        super().__init__(env)

        self.list_variable_name = list_variable_name
        # Action space setup
        low = np.array([env.action_space[v].low[0] for v in list_variable_name])
        high = np.array([env.action_space[v].high[0] for v in list_variable_name])
        self._action_space = Box(
            low=low, high=high, dtype=env.action_space[list_variable_name[0]].dtype
        )

        self.w_cell = w_cell

        # Paths and configuration
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
        self.mode = "train"
        self.type_mode = None
        self.dataset_name = "default"
        self.base_output_dir = (
            self.env.get_wrapper_attr("x_root").xpath("//save/folder")[0].text
        )

        os.makedirs(self.base_output_dir, exist_ok=True)
        self.list_data = []

        # Master seed from XML
        self.seed_val = int(
            self.env.get_wrapper_attr("x_root").xpath("//random_seed")[0].text
        )
        self.settingxml = self.env.get_wrapper_attr("settingxml")
        self.dt_gym = float(
            self.env.get_wrapper_attr("x_root")
            .xpath("//user_parameters/dt_gym")[0]
            .text
        )
        self.frequence_episode_test = frequence_episode_test

    @property
    def action_space(self):
        return self._action_space

    def change_xml(self, keys: list[str], elements: list):
        """Writes changes to the XML file on disk."""
        for key, element in zip(keys, elements):
            self.env.get_wrapper_attr("x_root").xpath(key)[0].text = str(element)
        self.env.get_wrapper_attr("x_tree").write(self.settingxml, pretty_print=True)

    def update_cell_path_cell_folder(self, path_cells_csv: str):
        p = Path(path_cells_csv)
        self.change_xml(
            keys=[
                "//initial_conditions/cell_positions/folder",
                "//initial_conditions/cell_positions/filename",
            ],
            elements=[str(p.parent), p.name],
        )
        self.csv_path_init = path_cells_csv
        self.cell_name_file = p.name
        self.cell_positions_folder = str(p.parent)

    def initial_condition_generation(self, generation_cfg=None):
        if self.generation_cfg is None:
            if generation_cfg is None:
                raise ValueError("generation_cfg must be provided at least once")

            self.generation_cfg = generation_cfg.copy()
            # Set bounds from env
            self.generation_cfg["x_min"] = self.env.unwrapped.x_min * 0.9
            self.generation_cfg["y_min"] = self.env.unwrapped.y_min * 0.9
            self.generation_cfg["x_max"] = self.env.unwrapped.x_max * 0.9
            self.generation_cfg["y_max"] = self.env.unwrapped.y_max * 0.9

            self.mode_train = self.generation_cfg.pop("mode_train")
            self.mode_test = self.generation_cfg.pop("mode_test")
            self.generation_cfg.setdefault("seed", self.seed_val)
            self.dataset_name = self.generation_cfg.get("dataset", "generated")

        self.generation_cfg["mode"] = (
            self.mode_train if self.mode == "train" else self.mode_test
        )

        ic_dir = os.path.join(
            self.base_output_dir, self.mode, "initial_conditions", self.dataset_name
        )
        os.makedirs(ic_dir, exist_ok=True)

        # Ensure unique seed per episode
        episode = self.env.unwrapped.episode + 1  # Use next episode ID
        csv_path = os.path.join(ic_dir, f"ic_{str(episode).zfill(6)}.csv")

        gen_cfg = self.generation_cfg.copy()
        # Combining master seed + episode for variation
        gen_cfg["seed"] = self.generation_cfg["seed"] + episode
        gen_cfg["csv_path"] = csv_path

        _, self.type_mode = generate_initial_condition(**gen_cfg)
        self.update_cell_path_cell_folder(csv_path)

    def reset(
        self,
        seed=None,
        options=None,
        generation_cfg=None,
        no_generation_cfg=None,
        **kwargs,
    ):
        # 1. Update Episode Counter and Mode
        # The unwrapped env usually increments episode inside its own reset or step
        # We look ahead to determine if this is a test episode
        next_episode = self.env.unwrapped.episode + 1
        self.mode = (
            "test" if (next_episode % self.frequence_episode_test == 0) else "train"
        )
        self.generate_physicell_data = True  # (self.mode == "test")

        if seed is not None:
            self.seed_val = seed

        # 2. Generate Initial Conditions BEFORE calling env.reset()
        if generation_cfg is not None or self.generation_cfg is not None:
            self.initial_condition_generation(generation_cfg=generation_cfg)

        if no_generation_cfg is not None or self.no_generation_cfg is not None:
            self.initial_condition(no_generation_cfg=no_generation_cfg)

        # 3. Save telemetry from previous run
        self.save_data()

        # 4. Prepare info for current run
        self.info = {"train_test": self.mode}

        # 5. Call parent reset (This is when PhysiCell actually starts and reads the updated XML)
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        d_action = {
            v: np.array([val]) for v, val in zip(self.list_variable_name, action)
        }

        max_radius = np.sqrt(
            (self.env.unwrapped.width / 2) ** 2 + (self.env.unwrapped.height / 2) ** 2
        )

        # 2. Scale the [0, 1] actions to physical PhysiCell dimensions
        # Dose is already [0, 1], so you just multiply by your maximum allowed concentration

        # Scale X and Y coordinates
        d_action["drug_1_x"] = self.env.unwrapped.x_min + (
            d_action["drug_1_x"] * self.env.unwrapped.width
        )
        d_action["drug_1_y"] = self.env.unwrapped.y_min + (
            d_action["drug_1_y"] * self.env.unwrapped.height
        )

        # Scale Radius
        d_action["drug_1_radius"] *= max_radius

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)
        dose_spent = self.env.unwrapped.get_wrapper_attr("get_dose_spent")()

        info.update(
            {
                "dose_spent": dose_spent,
                "type_mode": self.type_mode,
                "step_episode": self.env.unwrapped.step_episode,
                "train_test": self.mode,
            }
        )
        reward = self.w_cell * r_cancer_cells - dose_spent
        print(
            f"nb_tumor {info['number_tumor']} reward cancer cells {r_cancer_cells}_{self.w_cell * r_cancer_cells}, dose_spent {dose_spent}, total reward {reward}"
        )
        self.list_data.append(
            {
                "step": self.env.unwrapped.step_episode,
                "reward": reward,
                "dose_spent": dose_spent,
                "number_tumor": info.get("number_tumor", 0),
                "train_test": self.mode,
            }
        )

        return obs, reward, terminated, truncated, info

    def save_data(self):
        run_idx = self.env.unwrapped.episode
        if run_idx < 0 or not self.list_data:
            return

        out_dir = self._episode_output_dir(run_idx)
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(self.list_data)
        if "dose_spent" in df.columns:
            df["cumulative_dose_spent"] = df["dose_spent"].cumsum()
        if "reward" in df.columns:
            df["cumulative_reward"] = df["reward"].cumsum()

        df.to_csv(os.path.join(out_dir, "data.csv"), index=False)
        shutil.copy(
            self.csv_path_init,
            os.path.join(out_dir, os.path.basename(self.csv_path_init)),
        )

        self.list_data = []
        # Prep next save location in XML
        self.change_xml(
            keys=["//save/folder", "//save/SVG/enable"],
            elements=[out_dir, "true" if self.generate_physicell_data else "false"],
        )

    def _episode_output_dir(self, run_idx: int):
        return os.path.join(
            self.base_output_dir, self.mode, "episodes", f"run_{str(run_idx).zfill(6)}"
        )

    def initial_condition(self, no_generation_cfg=None):
        self.dataset_name = no_generation_cfg.get("dataset", "replay")
        if not hasattr(self, "list_csv"):
            self.list_csv = no_generation_cfg["list_csv"]
            self.current_csv_idx = 0

        csv_path = self.list_csv[self.current_csv_idx % len(self.list_csv)]
        self.current_csv_idx += 1
        self.update_cell_path_cell_folder(csv_path)
