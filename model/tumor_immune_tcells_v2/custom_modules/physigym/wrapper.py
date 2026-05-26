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
        list_variable_name: list[str] = ["drug_1_dose", "drug_1_x", "drug_1_y", "drug_1_radius"],
        w_cell=0.5,
        frequence_episode_test=3,
    ):
        """
        Wraps a PhysiCell environment to use a flat continuous Box action space.
        Ensures XML and CSV paths are updated before the simulation starts.

        type_mode tracking
        ──────────────────
        self.mode      : "train" | "test"   — set each reset()
        self.type_mode : str                — the exact geometry used this episode
                         e.g. "network_field", "circular", "rectangle"
                         Starts as "init" until the first generation runs.
        Test modes rotate deterministically through mode_test pool via
        self._test_mode_idx so every geometry gets equal coverage.
        """
        super().__init__(env)

        self.list_variable_name = list_variable_name

        # ── Action space ────────────────────────────────────────
        low  = np.array([env.action_space[v].low[0]  for v in list_variable_name])
        high = np.array([env.action_space[v].high[0] for v in list_variable_name])
        self._action_space = Box(
            low=low, high=high,
            dtype=env.action_space[list_variable_name[0]].dtype,
        )

        self.w_cell = w_cell

        # ── Paths ────────────────────────────────────────────────
        x_root = self.env.get_wrapper_attr("x_root")
        self.cell_positions_folder = x_root.xpath("//initial_conditions/cell_positions/folder")[0].text
        self.cell_name_file        = x_root.xpath("//initial_conditions/cell_positions/filename")[0].text
        self.csv_path_init         = os.path.join(self.cell_positions_folder, self.cell_name_file)
        self.base_output_dir       = x_root.xpath("//save/folder")[0].text
        self.settingxml            = self.env.get_wrapper_attr("settingxml")
        self.dt_gym                = float(x_root.xpath("//user_parameters/dt_gym")[0].text)
        self.seed_val              = int(x_root.xpath("//random_seed")[0].text)

        os.makedirs(self.base_output_dir, exist_ok=True)

        # ── Episode state ────────────────────────────────────────
        self.list_data            = []
        self.generation_cfg       = None   # set on first initial_condition_generation()
        self.no_generation_cfg    = None
        self.generate_physicell_data = False
        self.dataset_name         = "default"

        # ── Mode tracking ────────────────────────────────────────
        self.mode                 = "train"   # "train" | "test"
        self.type_mode            = "init"    # geometry used this episode
        self._test_mode_idx       = 0         # rotates through mode_test pool
        self.frequence_episode_test = frequence_episode_test

        # ── Mode-specific return buffers (window=50) ─────────────
        # Populated in reset() / used externally via wrapper attr
        self._return_buffers: dict[str, list] = {}

    # ── Properties ──────────────────────────────────────────────

    @property
    def action_space(self):
        return self._action_space

    # ── XML helpers ──────────────────────────────────────────────

    def change_xml(self, keys: list[str], elements: list):
        """Write key=value pairs to the XML file on disk."""
        x_root = self.env.get_wrapper_attr("x_root")
        x_tree = self.env.get_wrapper_attr("x_tree")
        for key, element in zip(keys, elements):
            x_root.xpath(key)[0].text = str(element)
        x_tree.write(self.settingxml, pretty_print=True)

    def update_cell_path_cell_folder(self, path_cells_csv: str):
        p = Path(path_cells_csv)
        self.change_xml(
            keys=[
                "//initial_conditions/cell_positions/folder",
                "//initial_conditions/cell_positions/filename",
            ],
            elements=[str(p.parent), p.name],
        )
        self.csv_path_init         = path_cells_csv
        self.cell_name_file        = p.name
        self.cell_positions_folder = str(p.parent)

    # ── Initial condition helpers ────────────────────────────────

    def initial_condition_generation(self, generation_cfg=None):
        """
        Generate a new initial condition CSV for the upcoming episode.

        First call: stores generation_cfg and extracts mode pools.
        Subsequent calls: reuses stored config, rotates through mode pools.

        Sets self.type_mode to the exact geometry string used.
        """
        # ── First-time setup ─────────────────────────────────────
        if self.generation_cfg is None:
            if generation_cfg is None:
                raise ValueError("generation_cfg must be provided at least once")

            self.generation_cfg = generation_cfg.copy()

            # Clamp to 90% of domain so cells don't spawn on walls
            self.generation_cfg["x_min"] = self.env.unwrapped.x_min * 0.9
            self.generation_cfg["y_min"] = self.env.unwrapped.y_min * 0.9
            self.generation_cfg["x_max"] = self.env.unwrapped.x_max * 0.9
            self.generation_cfg["y_max"] = self.env.unwrapped.y_max * 0.9

            # Pop mode pools — stored separately, not passed to generator directly
            raw_train = self.generation_cfg.pop("mode_train")
            raw_test  = self.generation_cfg.pop("mode_test")
            self.mode_train = raw_train if isinstance(raw_train, list) else [raw_train]
            self.mode_test  = raw_test  if isinstance(raw_test,  list) else [raw_test]

            self.generation_cfg.setdefault("seed", self.seed_val)
            self.dataset_name  = self.generation_cfg.get("dataset", "generated")
            self._test_mode_idx = 0

        # ── Pick geometry for this episode ───────────────────────
        if self.mode == "train":
            pool        = self.mode_train
            # Train rotates too — ensures all train geometries get coverage
            chosen_mode = pool[self._test_mode_idx % len(pool)]
            # only advance test index for test episodes
        else:
            pool        = self.mode_test
            chosen_mode = pool[self._test_mode_idx % len(pool)]
            self._test_mode_idx += 1   # advance only on test episodes

        self.generation_cfg["mode"] = chosen_mode

        # ── Write CSV ────────────────────────────────────────────
        ic_dir = os.path.join(
            self.base_output_dir, self.mode,
            "initial_conditions", self.dataset_name,
        )
        os.makedirs(ic_dir, exist_ok=True)

        episode  = self.env.unwrapped.episode + 1   # next episode id
        csv_path = os.path.join(ic_dir, f"ic_{str(episode).zfill(6)}.csv")

        gen_cfg             = self.generation_cfg.copy()
        gen_cfg["seed"]     = self.generation_cfg["seed"] + episode   # unique per episode
        gen_cfg["csv_path"] = csv_path

        _, self.type_mode = generate_initial_condition(**gen_cfg)
        # type_mode is now always a plain str e.g. "circular"
        self.update_cell_path_cell_folder(csv_path)

    def initial_condition(self, no_generation_cfg=None):
        """Replay mode: cycle through a fixed list of CSV files."""
        if no_generation_cfg is None:
            no_generation_cfg = self.no_generation_cfg
        self.dataset_name = no_generation_cfg.get("dataset", "replay")
        if not hasattr(self, "list_csv"):
            self.list_csv        = no_generation_cfg["list_csv"]
            self.current_csv_idx = 0
        csv_path = self.list_csv[self.current_csv_idx % len(self.list_csv)]
        self.current_csv_idx += 1
        self.type_mode = "replay"
        self.update_cell_path_cell_folder(csv_path)

    # ── Core gym interface ───────────────────────────────────────

    def reset(self, seed=None, options=None, generation_cfg=None, no_generation_cfg=None, **kwargs):
        """
        Reset flow:
        1. Determine train/test mode for the NEXT episode
        2. Generate initial conditions (updates self.type_mode)
        3. Save telemetry from the previous episode
        4. Call inner reset
        5. Inject wrapper keys into info so the actor process always sees them
        """
        # 1. Mode for next episode
        next_episode = self.env.unwrapped.episode + 1
        self.mode = "test" if (next_episode % self.frequence_episode_test == 0) else "train"
        self.generate_physicell_data = True if self.mode == "test" else False

        if seed is not None:
            self.seed_val = seed

        # 2. Generate initial conditions
        if generation_cfg is not None or self.generation_cfg is not None:
            self.initial_condition_generation(generation_cfg=generation_cfg)

        if no_generation_cfg is not None or self.no_generation_cfg is not None:
            self.initial_condition(no_generation_cfg=no_generation_cfg)

        # 3. Save previous episode telemetry
        self.save_data()

        # 4. Inner reset
        obs, info = self.env.reset(seed=seed, options=options)

        # 5. Always inject — actor process guard never silently drops again
        info["train_test"]   = self.mode
        info["type_mode"]    = self.type_mode   # always a str, never None
        info["step_episode"] = 0

        return obs, info

    def step(self, action: np.ndarray):
        # Build action dict
        d_action = {v: np.array([val]) for v, val in zip(self.list_variable_name, action)}

        max_radius = np.sqrt(
            (self.env.unwrapped.width  / 2) ** 2 +
            (self.env.unwrapped.height / 2) ** 2
        )

        # Scale [0,1] actions → physical PhysiCell coordinates
        d_action["drug_1_x"]      = self.env.unwrapped.x_min + d_action["drug_1_x"]      * self.env.unwrapped.width
        d_action["drug_1_y"]      = self.env.unwrapped.y_min + d_action["drug_1_y"]      * self.env.unwrapped.height
        d_action["drug_1_radius"] = d_action["drug_1_radius"] * max_radius

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)
        dose_spent = self.env.unwrapped.get_wrapper_attr("get_dose_spent")()

        # Always inject wrapper keys — same pattern as reset()
        info.update({
            "dose_spent":   dose_spent,
            "type_mode":    self.type_mode,
            "step_episode": self.env.unwrapped.step_episode,
            "train_test":   self.mode,
        })

        reward = self.w_cell * r_cancer_cells - dose_spent

        self.list_data.append({
            "step":         self.env.unwrapped.step_episode,
            "reward":       reward,
            "dose_spent":   dose_spent,
            "number_tumor": info.get("number_tumor", 0),
            "train_test":   self.mode,
            "type_mode":    self.type_mode,
        })

        return obs, reward, terminated, truncated, info

    # ── Telemetry ────────────────────────────────────────────────

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

        # Save CSV and initial condition
        df.to_csv(os.path.join(out_dir, "data.csv"), index=False)
        shutil.copy(
            self.csv_path_init,
            os.path.join(out_dir, os.path.basename(self.csv_path_init)),
        )

        # ---------------------------------------------------------
        # PNG PLOT GENERATION (Only if we are in testing mode)
        # ---------------------------------------------------------
        if self.generate_physicell_data:
            import matplotlib.pyplot as plt
            
            fig, ax1 = plt.subplots(figsize=(8, 5))
            
            # Example plot: Reward over time
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Reward", color="tab:blue")
            ax1.plot(df["step"], df["reward"], color="tab:blue", label="Reward")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            # Example secondary plot: Cumulative Dose
            if "cumulative_dose_spent" in df.columns:
                ax2 = ax1.twinx()
                ax2.set_ylabel("Cumulative Dose", color="tab:red")
                ax2.plot(df["step"], df["cumulative_dose_spent"], color="tab:red", linestyle="--")
                ax2.tick_params(axis="y", labelcolor="tab:red")

            plt.title(f"Episode {run_idx} Telemetry ({self.type_mode})")
            fig.tight_layout()
            
            # Save as PNG
            plt.savefig(os.path.join(out_dir, "episode_metrics.png"), dpi=150)
            plt.close(fig)
        # ---------------------------------------------------------

        self.list_data = []

        # Point PhysiCell output to the episode folder for the NEXT run
        # Toggle BOTH SVG and full_data flags to strictly silence training runs
        is_active = "true" if self.generate_physicell_data else "false"
        
        self.change_xml(
            keys=[
                "//save/folder", 
                "//save/SVG/enable", 
                "//save/full_data/enable"  # <-- Added to silence .mat files during training
            ],
            elements=[
                out_dir, 
                is_active, 
                is_active                  # <-- Added
            ],
        )

    def _episode_output_dir(self, run_idx: int) -> str:
        return os.path.join(
            self.base_output_dir, self.mode,
            "episodes", f"run_{str(run_idx).zfill(6)}",
        )