import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os
import pandas as pd
import shutil
import subprocess
import tempfile
from init_conds_v3 import generate_initial_condition
from pathlib import Path


# ── per-substrate display config ─────────────────────────────────────────────
# (label, colormap)  — extend if substrates change
_SUBSTRATE_STYLE = {
    "anti_tumoral_factor": ("Anti-tumoral",  "Greens"),
    "pro_tumoral_factor":  ("Pro-tumoral",   "Reds"),
    "drug_1":              ("Drug",          "Purples"),
    "tumor_molecule":      ("Tumor mol.",    "Oranges"),
    "cytokine":            ("Cytokine",      "Blues"),
}
_SUBSTRATE_STYLE_DEFAULT = ("Blues",)   # fallback cmap


def _render_frame(
    cells_img,          # uint8 (n_types, H, W)
    subs_img,           # uint8 (n_subs,  H, W)
    cell_type_names,    # list[str] in id order
    cell_type_colors,   # dict name→hex/color
    substrate_names,    # list[str] in id order
    action,             # np.ndarray [dose, x_norm, y_norm, radius_norm]
    step,
    episode,
    type_mode,
    reward_history,     # list of per-step rewards up to this step
    dose_history,       # list of per-step doses up to this step
):
    """
    Renders one video frame as a matplotlib figure and returns a numpy RGB array.

    Layout (left → right):
      col 0        : cell scatter (one heatmap per type, stacked vertically)
      col 1..n_subs: substrate heatmaps (one per substrate, stacked vertically)
      col -1       : telemetry panel (reward curve + dose bar)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import to_rgba

    n_types = cells_img.shape[0]
    n_subs  = subs_img.shape[0]
    H, W    = cells_img.shape[1], cells_img.shape[2]

    # ── figure layout ─────────────────────────────────────────────
    # columns: [cells | subs... | telemetry]
    n_cols   = 1 + n_subs + 1
    fig_w    = 2.2 * n_cols
    fig_h    = max(2.2 * max(n_types, n_subs), 4.0)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")

    # outer grid: [cells-col | subs-cols... | telemetry-col]
    outer = gridspec.GridSpec(
        1, n_cols,
        figure=fig,
        left=0.03, right=0.97,
        top=0.88,  bottom=0.05,
        wspace=0.12,
    )

    # ── header ────────────────────────────────────────────────────
    fig.text(
        0.5, 0.95,
        f"Episode {episode:06d}  |  Step {step:03d}  |  {type_mode}",
        ha="center", va="top", fontsize=9, color="white",
        fontweight="bold",
    )

    # ── helpers ───────────────────────────────────────────────────
    def _axis_off(ax, title, title_color="white"):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.set_title(title, fontsize=7, color=title_color, pad=2)

    # ── col 0: cell heatmaps ──────────────────────────────────────
    cell_inner = gridspec.GridSpecFromSubplotSpec(
        n_types, 1, subplot_spec=outer[0], hspace=0.08
    )
    dose_norm   = float(action[0]) if len(action) > 0 else 0.0
    x_norm      = float(action[1]) if len(action) > 1 else 0.5
    y_norm      = float(action[2]) if len(action) > 2 else 0.5
    radius_norm = float(action[3]) if len(action) > 3 else 0.0

    # injection circle in pixel coords
    cx_px = x_norm * W
    cy_px = (1.0 - y_norm) * H   # flip y: image row 0 = top
    r_px  = radius_norm * min(H, W)

    for i, name in enumerate(cell_type_names):
        ax = fig.add_subplot(cell_inner[i])
        channel = cells_img[i].astype(float)

        # build a single-hue RGBA image tinted by cell_type_color
        color = to_rgba(cell_type_colors.get(name, "gray"))
        rgba  = np.zeros((H, W, 4), dtype=float)
        alpha_channel = channel / 255.0
        for c in range(3):
            rgba[:, :, c] = color[c]
        rgba[:, :, 3] = alpha_channel
        # dark background
        bg = np.full((H, W, 4), [0.08, 0.08, 0.15, 1.0])
        # composite over dark bg
        out = bg.copy()
        a = rgba[:, :, 3:4]
        out[:, :, :3] = rgba[:, :, :3] * a + bg[:, :, :3] * (1 - a)
        out[:, :, 3]  = 1.0

        ax.imshow(out, origin="upper", interpolation="nearest", aspect="equal")

        # draw injection circle on every cell panel
        if dose_norm > 0.01 and r_px > 0:
            circ = mpatches.Circle(
                (cx_px, cy_px), r_px,
                linewidth=1.2, edgecolor="white", facecolor="none",
                linestyle="--", alpha=0.7,
            )
            ax.add_patch(circ)
            ax.plot(cx_px, cy_px, "+", color="white", markersize=5, markeredgewidth=1.0)

        _axis_off(ax, name, title_color=cell_type_colors.get(name, "white"))

    # ── col 1..n_subs: substrate heatmaps ─────────────────────────
    for j, sname in enumerate(substrate_names):
        subs_inner = gridspec.GridSpecFromSubplotSpec(
            n_subs, 1, subplot_spec=outer[1 + j], hspace=0.08
        )
        # only draw the matching substrate in this column
        ax = fig.add_subplot(subs_inner[j])
        label, cmap = _SUBSTRATE_STYLE.get(sname, (sname, "viridis"))
        ax.imshow(
            subs_img[j],
            cmap=cmap, vmin=0, vmax=255,
            origin="upper", interpolation="nearest", aspect="equal",
        )
        # injection marker on drug channel only
        if sname == "drug_1" and dose_norm > 0.01 and r_px > 0:
            circ = mpatches.Circle(
                (cx_px, cy_px), r_px,
                linewidth=1.5, edgecolor="yellow", facecolor="none",
                linestyle="-", alpha=0.9,
            )
            ax.add_patch(circ)
            ax.plot(cx_px, cy_px, "+", color="yellow", markersize=6, markeredgewidth=1.2)
        _axis_off(ax, label)
        # blank the other rows in this column
        for k in range(n_subs):
            if k != j:
                ax_blank = fig.add_subplot(subs_inner[k])
                ax_blank.set_visible(False)

    # ── last col: telemetry ────────────────────────────────────────
    tel_inner = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[n_cols - 1], hspace=0.4
    )

    # reward curve
    ax_r = fig.add_subplot(tel_inner[0])
    ax_r.set_facecolor("#0d0d1a")
    if reward_history:
        ax_r.plot(reward_history, color="#00d4ff", linewidth=1.0)
    ax_r.axhline(0, color="#444466", linewidth=0.6, linestyle="--")
    ax_r.set_title("Reward", fontsize=7, color="white", pad=2)
    ax_r.tick_params(labelsize=5, colors="gray")
    for sp in ax_r.spines.values():
        sp.set_edgecolor("#444466")

    # cumulative reward
    ax_cr = fig.add_subplot(tel_inner[1])
    ax_cr.set_facecolor("#0d0d1a")
    if reward_history:
        ax_cr.plot(np.cumsum(reward_history), color="#a8ff78", linewidth=1.0)
    ax_cr.set_title("Cum. reward", fontsize=7, color="white", pad=2)
    ax_cr.tick_params(labelsize=5, colors="gray")
    for sp in ax_cr.spines.values():
        sp.set_edgecolor("#444466")

    # dose bar (current step)
    ax_d = fig.add_subplot(tel_inner[2])
    ax_d.set_facecolor("#0d0d1a")
    ax_d.barh(0, dose_norm, color="#ff6b6b", height=0.6)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(-0.5, 0.5)
    ax_d.set_title(f"Dose  {dose_norm:.2f}", fontsize=7, color="white", pad=2)
    ax_d.set_yticks([])
    ax_d.tick_params(labelsize=5, colors="gray")
    for sp in ax_d.spines.values():
        sp.set_edgecolor("#444466")

    # ── rasterise to numpy RGB ─────────────────────────────────────
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


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

        Video generation
        ────────────────
        On test episodes, each step() captures cells+substrates grids and the
        action into self._frame_buffer.  save_data() renders all frames to PNG
        in a temp dir then calls ffmpeg to compile video.mp4.
        PhysiCell SVG/full_data output is permanently disabled — all visual
        output comes exclusively from the observation arrays.
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
        self._frame_buffer        = []   # list of dicts: {cells, subs, action, reward, dose, n_tumor}
        self.generation_cfg       = None
        self.no_generation_cfg    = None
        self.generate_physicell_data = False
        self.dataset_name         = "default"

        # ── Mode tracking ────────────────────────────────────────
        self.mode                 = "train"
        self.type_mode            = "init"
        self._test_mode_idx       = 0
        self.frequence_episode_test = frequence_episode_test

        # ── Mode-specific return buffers (window=50) ─────────────
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
        if self.generation_cfg is None:
            if generation_cfg is None:
                raise ValueError("generation_cfg must be provided at least once")

            self.generation_cfg = generation_cfg.copy()

            self.generation_cfg["x_min"] = self.env.unwrapped.x_min * 0.9
            self.generation_cfg["y_min"] = self.env.unwrapped.y_min * 0.9
            self.generation_cfg["x_max"] = self.env.unwrapped.x_max * 0.9
            self.generation_cfg["y_max"] = self.env.unwrapped.y_max * 0.9

            raw_train = self.generation_cfg.pop("mode_train")
            raw_test  = self.generation_cfg.pop("mode_test")
            self.mode_train = raw_train if isinstance(raw_train, list) else [raw_train]
            self.mode_test  = raw_test  if isinstance(raw_test,  list) else [raw_test]

            self.generation_cfg.setdefault("seed", self.seed_val)
            self.dataset_name  = self.generation_cfg.get("dataset", "generated")
            self._test_mode_idx = 0

        if self.mode == "train":
            pool        = self.mode_train
            chosen_mode = pool[self._test_mode_idx % len(pool)]
        else:
            pool        = self.mode_test
            chosen_mode = pool[self._test_mode_idx % len(pool)]
            self._test_mode_idx += 1

        self.generation_cfg["mode"] = chosen_mode

        ic_dir = os.path.join(
            self.base_output_dir, self.mode,
            "initial_conditions", self.dataset_name,
        )
        os.makedirs(ic_dir, exist_ok=True)

        episode  = self.env.unwrapped.episode + 1
        csv_path = os.path.join(ic_dir, f"ic_{str(episode).zfill(6)}.csv")

        gen_cfg             = self.generation_cfg.copy()
        gen_cfg["seed"]     = self.generation_cfg["seed"] + episode
        gen_cfg["csv_path"] = csv_path

        _, self.type_mode = generate_initial_condition(**gen_cfg)
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
        3. Save telemetry + video from the previous episode
        4. Call inner reset
        5. Inject wrapper keys into info
        """
        next_episode = self.env.unwrapped.episode + 1
        self.mode = "test" if (next_episode % self.frequence_episode_test == 0) else "train"
        self.generate_physicell_data = (self.mode == "test")

        if seed is not None:
            self.seed_val = seed

        if generation_cfg is not None or self.generation_cfg is not None:
            self.initial_condition_generation(generation_cfg=generation_cfg)

        if no_generation_cfg is not None or self.no_generation_cfg is not None:
            self.initial_condition(no_generation_cfg=no_generation_cfg)

        self.save_data()

        obs, info = self.env.reset(seed=seed, options=options)

        info["train_test"]   = self.mode
        info["type_mode"]    = self.type_mode
        info["step_episode"] = 0

        return obs, info

    def step(self, action: np.ndarray):
        d_action = {v: np.array([val]) for v, val in zip(self.list_variable_name, action)}

        max_radius = np.sqrt(
            (self.env.unwrapped.width  / 2) ** 2 +
            (self.env.unwrapped.height / 2) ** 2
        )

        d_action["drug_1_x"]      = self.env.unwrapped.x_min + d_action["drug_1_x"]      * self.env.unwrapped.width
        d_action["drug_1_y"]      = self.env.unwrapped.y_min + d_action["drug_1_y"]      * self.env.unwrapped.height
        d_action["drug_1_radius"] = d_action["drug_1_radius"] * max_radius

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)
        dose_spent = self.env.unwrapped.get_wrapper_attr("get_dose_spent")()

        info.update({
            "dose_spent":   dose_spent,
            "type_mode":    self.type_mode,
            "step_episode": self.env.unwrapped.step_episode,
            "train_test":   self.mode,
        })

        reward = self.w_cell * r_cancer_cells - dose_spent

        row = {
            "step":         self.env.unwrapped.step_episode,
            "reward":       reward,
            "dose_spent":   dose_spent,
            "number_tumor": info.get("number_tumor", 0),
            "train_test":   self.mode,
            "type_mode":    self.type_mode,
        }
        self.list_data.append(row)

        # capture spatial frame for test episodes only
        if self.generate_physicell_data:
            env_inner = self.env.unwrapped
            self._frame_buffer.append({
                "cells":   env_inner.get_matrix_cells().copy(),    # uint8 (n_types, H, W)
                "subs":    env_inner.get_matrix_substrates().copy(), # uint8 (n_subs,  H, W)
                "action":  action.copy(),                           # raw [0,1] values
                "reward":  reward,
                "dose":    dose_spent,
                "n_tumor": info.get("number_tumor", 0),
            })

        return obs, reward, terminated, truncated, info

    # ── Telemetry & video ────────────────────────────────────────

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

        if self.generate_physicell_data and self._frame_buffer:
            self._compile_video(out_dir, run_idx)

        self.list_data     = []
        self._frame_buffer = []

        # PhysiCell never writes SVG or .mat files — all output comes from wrapper
        self.change_xml(
            keys=[
                "//save/folder",
                "//save/SVG/enable",
                "//save/full_data/enable",
            ],
            elements=[out_dir, "false", "false"],
        )

    def _compile_video(self, out_dir: str, run_idx: int):
        """Render all buffered frames to PNG then compile to video.mp4 via ffmpeg."""
        env_inner        = self.env.unwrapped
        cell_type_names  = list(env_inner.cell_type_to_id.keys())
        cell_type_colors = env_inner.cell_type_to_color
        substrate_names  = list(env_inner.substrate_unique)
        episode          = run_idx

        reward_so_far = []
        dose_so_far   = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, frame in enumerate(self._frame_buffer):
                reward_so_far.append(frame["reward"])
                dose_so_far.append(frame["dose"])

                rgb = _render_frame(
                    cells_img       = frame["cells"],
                    subs_img        = frame["subs"],
                    cell_type_names = cell_type_names,
                    cell_type_colors= cell_type_colors,
                    substrate_names = substrate_names,
                    action          = frame["action"],
                    step            = i,
                    episode         = episode,
                    type_mode       = self.type_mode,
                    reward_history  = list(reward_so_far),
                    dose_history    = list(dose_so_far),
                )

                # save as PNG with zero-padded index for ffmpeg glob
                import cv2
                png_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
                cv2.imwrite(png_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            video_path = os.path.join(out_dir, "video.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", "10",
                    "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-threads", "1",
                    video_path,
                ],
                check=True,
                capture_output=True,
            )

    def _episode_output_dir(self, run_idx: int) -> str:
        return os.path.join(
            self.base_output_dir, self.mode,
            "episodes", f"run_{str(run_idx).zfill(6)}",
        )
