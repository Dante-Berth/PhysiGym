import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os
import pandas as pd
import shutil
from init_conds import generate_initial_condition
from pathlib import Path


# ── per-substrate display config ─────────────────────────────────────────────
# (label, colormap)  — extend if substrates change
_SUBSTRATE_STYLE = {
    "anti_tumoral_factor": ("Anti-tumoral", "Greens"),
    "pro_tumoral_factor": ("Pro-tumoral", "Reds"),
    "drug_1": ("Drug", "Purples"),
    "tumor_molecule": ("Tumor mol.", "Oranges"),
    "cytokine": ("Cytokine", "Blues"),
}
_SUBSTRATE_STYLE_DEFAULT = ("Blues",)  # fallback cmap


def _render_frame(
    cells_img,  # uint8 (n_types, H, W)
    subs_img,  # uint8 (n_subs,  H, W)
    cell_type_names,  # list[str] in id order
    cell_type_colors,  # dict name→hex/color
    substrate_names,  # list[str] in id order
    action,  # np.ndarray [dose, x_norm, y_norm, radius_norm]
    step,
    episode,
    type_mode,
    reward_history,  # list of per-step rewards up to this step
    dose_history,  # list of per-step doses up to this step
    x_min=0.0,
    x_max=1.0,  # env.unwrapped.x_min / x_max (physical units)
    y_min=0.0,
    y_max=1.0,  # env.unwrapped.y_min / y_max (physical units)
    inj_x=None,  # physical injection center x as applied to the sim
    inj_y=None,  # physical injection center y as applied to the sim
    inj_radius=None,  # physical injection radius as applied to the sim
):
    """
    Renders one video frame as a matplotlib figure and returns a numpy RGB array.

    Layout:
      Top row   : [composite cells + drug circle (large, square)] | [reward curve / cum-dose panel]
      Bottom row: substrate heatmaps side by side (one per substrate)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import to_rgba

    n_subs = subs_img.shape[0]
    H, W = cells_img.shape[1], cells_img.shape[2]

    domain_width = x_max - x_min
    domain_height = y_max - y_min

    dose_norm = float(action[0]) if len(action) > 0 else 0.0
    x_norm = float(action[1]) if len(action) > 1 else 0.5
    y_norm = float(action[2]) if len(action) > 2 else 0.5
    radius_norm = float(action[3]) if len(action) > 3 else 0.0

    # injection circle in physical coords.
    # Prefer the geometry actually applied to the sim (passed in from the frame
    # buffer) so the overlay matches the drug substrate. Only fall back to
    # re-deriving from the raw action for legacy npz files that lack it — note
    # that fallback is approximate and will NOT match the wrapper's 5–20%
    # half-diagonal radius mapping.
    if (
        inj_x is not None
        and inj_y is not None
        and inj_radius is not None
        and not (np.isnan(inj_x) or np.isnan(inj_y) or np.isnan(inj_radius))
    ):
        cx_phys = inj_x
        cy_phys = inj_y
        r_phys = inj_radius
    else:
        cx_phys = x_min + x_norm * domain_width
        cy_phys = y_min + y_norm * domain_height
        r_phys = radius_norm * max(domain_width, domain_height)

    # ── composite: blend all cell types onto white background ────────
    # Fixed colors regardless of what the env colormap assigned:
    _CELL_COLORS = {"tumor": "gray", "t_cell": "red", "macrophage": "yellow"}

    bg_rgb = np.array([1.0, 1.0, 1.0])  # white
    composite = np.full((H, W, 3), bg_rgb)
    for i, name in enumerate(cell_type_names):
        hex_color = _CELL_COLORS.get(name, cell_type_colors.get(name, "gray"))
        color = np.array(to_rgba(hex_color)[:3])
        # get_matrix stores slices as [x_bin, y_bin]; imshow reads axis-0 as the
        # vertical (y) axis, so transpose to [y_bin, x_bin] to align the image
        # with physical (x, y) — otherwise cells/drug appear reflected across the
        # diagonal relative to the injection-circle overlay drawn at (cx, cy).
        alpha = cells_img[i].astype(float).T / 255.0  # (y, x)
        for c in range(3):
            composite[:, :, c] = composite[:, :, c] * (1 - alpha) + color[c] * alpha
    composite = np.clip(composite, 0, 1)

    # ── figure ────────────────────────────────────────────────────
    # cell panel sized to match physical domain aspect ratio
    aspect = domain_height / max(domain_width, 1e-8)  # h/w ratio
    cell_panel_w = 3.8
    cell_panel_h = cell_panel_w * aspect
    telem_w = 2.6
    sub_h = 1.1  # fixed height for substrate row
    fig_w = cell_panel_w + telem_w + 0.4
    fig_h = cell_panel_h + sub_h + 0.6  # top + bottom + header

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)
    fig.patch.set_facecolor("#1a1a2e")

    # outer: 2 rows × 2 cols; bottom row spans both cols
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        left=0.03,
        right=0.97,
        top=0.91,
        bottom=0.04,
        hspace=0.28,
        wspace=0.12,
        height_ratios=[cell_panel_h, sub_h],
        width_ratios=[cell_panel_w, telem_w],
    )

    # ── header ────────────────────────────────────────────────────
    fig.text(
        0.5,
        0.97,
        f"Episode {episode:06d}  |  Step {step:03d}  |  {type_mode}",
        ha="center",
        va="top",
        fontsize=9,
        color="white",
        fontweight="bold",
    )

    # ── helpers ───────────────────────────────────────────────────
    def _style_ax(ax, title, title_color="white"):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.set_title(title, fontsize=7, color=title_color, pad=2)

    def _style_plot_ax(ax, title):
        ax.set_facecolor("#0d0d1a")
        ax.set_title(title, fontsize=7, color="white", pad=2)
        ax.tick_params(labelsize=5, colors="gray")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

    # ── top-left: composite cell panel ────────────────────────────
    # extent = [left, right, bottom, top] in physical units
    cell_extent = [x_min, x_max, y_min, y_max]
    ax_cells = fig.add_subplot(outer[0, 0])
    ax_cells.imshow(
        composite,
        origin="lower",
        extent=cell_extent,
        interpolation="nearest",
        aspect="equal",
    )
    ax_cells.set_xlim(x_min, x_max)
    ax_cells.set_ylim(y_min, y_max)

    # draw per-type legend
    legend_handles = []
    for name in cell_type_names:
        c = _CELL_COLORS.get(name, cell_type_colors.get(name, "gray"))
        legend_handles.append(mpatches.Patch(color=c, label=name))
    ax_cells.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=5,
        framealpha=0.7,
        facecolor="white",
        edgecolor="#aaaaaa",
        labelcolor="black",
    )

    # dose value top-right
    ax_cells.text(
        0.98,
        0.98,
        f"dose {dose_norm:.3f}",
        transform=ax_cells.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="#0044cc",
        fontweight="bold",
    )

    # injection circle: filled with alpha ∝ dose, dashed outline always visible
    if r_phys > 0:
        # filled area encodes dose intensity
        fill = mpatches.Circle(
            (cx_phys, cy_phys),
            r_phys,
            linewidth=0,
            edgecolor="none",
            facecolor="#0044cc",
            alpha=dose_norm * 0.5,
        )
        ax_cells.add_patch(fill)
        # dashed outline + crosshair only when dose > threshold
        if dose_norm > 0.01:
            outline = mpatches.Circle(
                (cx_phys, cy_phys),
                r_phys,
                linewidth=1.8,
                edgecolor="#0044cc",
                facecolor="none",
                linestyle="--",
                alpha=0.9,
            )
            ax_cells.add_patch(outline)
            ax_cells.plot(
                cx_phys,
                cy_phys,
                "+",
                color="#0044cc",
                markersize=7,
                markeredgewidth=1.5,
            )

    _style_ax(ax_cells, "Cells (composite)", title_color="white")

    # ── top-right: telemetry ──────────────────────────────────────
    tel_inner = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[0, 1], hspace=0.55
    )

    ax_r = fig.add_subplot(tel_inner[0])
    _style_plot_ax(ax_r, "Reward / step")
    if reward_history:
        ax_r.plot(reward_history, color="#00d4ff", linewidth=1.0)
    ax_r.axhline(0, color="#444466", linewidth=0.6, linestyle="--")

    ax_cr = fig.add_subplot(tel_inner[1])
    _style_plot_ax(ax_cr, "Cumulative reward")
    if reward_history:
        ax_cr.plot(np.cumsum(reward_history), color="#a8ff78", linewidth=1.0)
    ax_cr.axhline(0, color="#444466", linewidth=0.6, linestyle="--")

    ax_d = fig.add_subplot(tel_inner[2])
    _style_plot_ax(ax_d, f"Cumulative dose  {sum(dose_history):.2f}")
    if dose_history:
        ax_d.plot(np.cumsum(dose_history), color="#ff6b6b", linewidth=1.0)
    ax_d.axhline(0, color="#444466", linewidth=0.6, linestyle="--")

    # ── bottom row: substrate heatmaps (spans both columns) ───────
    # merge bottom row into one wide subplot spec, then subdivide
    bottom_spec = gridspec.GridSpecFromSubplotSpec(
        1,
        n_subs,
        subplot_spec=outer[1, :],  # span both columns
        wspace=0.12,
    )
    sub_extent = [x_min, x_max, y_min, y_max]
    for j, sname in enumerate(substrate_names):
        ax_s = fig.add_subplot(bottom_spec[j])
        label, cmap = _SUBSTRATE_STYLE.get(sname, (sname, "viridis"))
        ax_s.imshow(
            subs_img[j].T,  # [x_bin, y_bin] → [y_bin, x_bin] to match physical (x, y)
            cmap=cmap,
            vmin=0,
            vmax=255,
            origin="lower",
            extent=sub_extent,
            interpolation="nearest",
            aspect="equal",
        )
        ax_s.set_xlim(x_min, x_max)
        ax_s.set_ylim(y_min, y_max)
        if sname == "drug_1" and dose_norm > 0.01 and r_phys > 0:
            circ = mpatches.Circle(
                (cx_phys, cy_phys),
                r_phys,
                linewidth=1.5,
                edgecolor="yellow",
                facecolor="none",
                linestyle="-",
                alpha=0.9,
            )
            ax_s.add_patch(circ)
            ax_s.plot(
                cx_phys, cy_phys, "+", color="yellow", markersize=5, markeredgewidth=1.2
            )
        _style_ax(ax_s, label)

    # ── rasterise to numpy RGB ─────────────────────────────────────
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    buf = buf[:, :, :3].copy()
    plt.close(fig)
    return buf


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
        w_dose=1.0,
        w_smooth=0.0,
        action_delta_max=None,
        frequence_episode_test=3,
        action_mode: str = "targeted",
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
        action into self._frame_buffer. At episode end save_data() renders those
        frames to PNGs under out_dir/frames/ (kept on disk); video_maker.py then
        compiles them into video.mp4 as a post-processing step.
        PhysiCell SVG/full_data output is permanently disabled — all visual
        output comes exclusively from the observation arrays.
        """
        super().__init__(env)

        self.list_variable_name = list_variable_name
        self.action_mode = action_mode

        # ── Action space ────────────────────────────────────────
        low = np.array([env.action_space[v].low[0] for v in list_variable_name])
        high = np.array([env.action_space[v].high[0] for v in list_variable_name])
        self._action_space = Box(
            low=low,
            high=high,
            dtype=env.action_space[list_variable_name[0]].dtype,
        )

        self.w_cell = w_cell
        self.w_dose = w_dose
        self.w_smooth = w_smooth
        # per-component max step in normalised [0,1] space; None = unconstrained
        # for "targeted" mode components are [dose, x_norm, y_norm, radius_norm]
        if action_delta_max is not None:
            self._action_delta_max = np.asarray(action_delta_max, dtype=np.float32)
        else:
            self._action_delta_max = None

        # ── Paths ────────────────────────────────────────────────
        x_root = self.env.get_wrapper_attr("x_root")
        self.cell_positions_folder = x_root.xpath(
            "//initial_conditions/cell_positions/folder"
        )[0].text
        self.cell_name_file = x_root.xpath(
            "//initial_conditions/cell_positions/filename"
        )[0].text
        self.csv_path_init = os.path.join(
            self.cell_positions_folder, self.cell_name_file
        )
        self.base_output_dir = x_root.xpath("//save/folder")[0].text
        self.settingxml = self.env.get_wrapper_attr("settingxml")
        self.dt_gym = float(x_root.xpath("//user_parameters/dt_gym")[0].text)
        self.seed_val = int(x_root.xpath("//random_seed")[0].text)

        os.makedirs(self.base_output_dir, exist_ok=True)

        # Disable PhysiCell file output permanently — must be set before first reset()
        self.change_xml(
            keys=["//save/SVG/enable", "//save/full_data/enable"],
            elements=["false", "false"],
        )

        # ── Episode state ────────────────────────────────────────
        self.list_data = []
        self._frame_buffer = []  # list of dicts: {cells, subs, action, reward, dose, n_tumor}
        self._action_history = []  # list of np.ndarray, one per env step in current episode
        self._decision_anchor = (
            None  # clipped action at start of current decision block
        )
        self._last_raw_decision = None  # raw action of the current decision block
        self._q_calib_buffer = []  # list of dicts: {obs, action, reward} for test episodes
        # IC + mode of the episode CURRENTLY running, captured at generation time
        # so save_data() finalizes the finished episode with its own IC/mode,
        # not the next episode's (which generation overwrites before save runs).
        self._running_csv_init = None
        self._running_mode = None
        self._running_type_mode = None
        self.generation_cfg = None
        self.no_generation_cfg = None
        self.generate_physicell_data = False
        self.emit_reward_analysis_rows = False
        self.dataset_name = "default"

        # ── Mode tracking ────────────────────────────────────────
        self.mode = "train"
        self.type_mode = "init"
        self._test_mode_idx = 0
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
        self.csv_path_init = path_cells_csv
        self.cell_name_file = p.name
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
            raw_test = self.generation_cfg.pop("mode_test")
            self.mode_train = raw_train if isinstance(raw_train, list) else [raw_train]
            self.mode_test = raw_test if isinstance(raw_test, list) else [raw_test]

            self.generation_cfg.setdefault("seed", self.seed_val)
            self.dataset_name = self.generation_cfg.get("dataset", "generated")
            self._test_mode_idx = 0

        if self.mode == "train":
            pool = self.mode_train
            chosen_mode = pool[self._test_mode_idx % len(pool)]
        else:
            pool = self.mode_test
            chosen_mode = pool[self._test_mode_idx % len(pool)]
            self._test_mode_idx += 1

        self.generation_cfg["mode"] = chosen_mode

        ic_dir = os.path.join(
            self.base_output_dir,
            self.mode,
            "initial_conditions",
            self.dataset_name,
        )
        os.makedirs(ic_dir, exist_ok=True)

        episode = self.env.unwrapped.episode + 1
        csv_path = os.path.join(ic_dir, f"ic_{str(episode).zfill(6)}.csv")

        gen_cfg = self.generation_cfg.copy()
        gen_cfg["seed"] = self.generation_cfg["seed"] + episode
        gen_cfg["csv_path"] = csv_path

        _, self.type_mode = generate_initial_condition(**gen_cfg)
        self.update_cell_path_cell_folder(csv_path)

    def initial_condition(self, no_generation_cfg=None):
        """Replay mode: cycle through a fixed list of CSV files."""
        if no_generation_cfg is None:
            no_generation_cfg = self.no_generation_cfg
        self.dataset_name = no_generation_cfg.get("dataset", "replay")
        if not hasattr(self, "list_csv"):
            self.list_csv = no_generation_cfg["list_csv"]
            self.current_csv_idx = 0
        csv_path = self.list_csv[self.current_csv_idx % len(self.list_csv)]
        self.current_csv_idx += 1
        self.type_mode = "replay"
        self.update_cell_path_cell_folder(csv_path)

    # ── Core gym interface ───────────────────────────────────────

    def reset(
        self,
        seed=None,
        options=None,
        generation_cfg=None,
        no_generation_cfg=None,
        **kwargs,
    ):
        """
        Reset flow:
        1. Finalize the PREVIOUS episode (uses its captured mode/IC, not self.mode)
        2. Determine train/test mode for the NEXT episode
        3. Generate initial conditions (updates self.type_mode / csv_path_init)
        4. Capture the NEXT episode's mode/IC as "running" for the next save
        5. Point //save/folder at the NEXT episode's own dir before inner reset
        6. Call inner reset
        7. Inject wrapper keys into info
        """
        if seed is not None:
            self.seed_val = seed

        # 1. Finalize the PREVIOUS episode using ITS OWN captured mode/IC.
        #    (self.mode / self.csv_path_init are about to be overwritten for the
        #    next episode, so save_data must not rely on them.)
        self.save_data()

        # 2. Decide mode for the next episode
        next_episode = self.env.unwrapped.episode + 1
        self.mode = (
            "test" if (next_episode % self.frequence_episode_test == 0) else "train"
        )
        # During reward-analysis / hyperparameter-search rollouts we only want the
        # CSV reward rows — suppress per-test-episode spatial frame capture and the
        # ffmpeg video.mp4 compile (which also writes mat/svg/html) to save disk.
        self.generate_physicell_data = self.mode == "test" and not getattr(
            self, "emit_reward_analysis_rows", False
        )

        # 3. Initial condition generation now sees the correct mode.
        #    This updates self.csv_path_init and self.type_mode for the next episode.
        if generation_cfg is not None or self.generation_cfg is not None:
            self.initial_condition_generation(generation_cfg=generation_cfg)

        if no_generation_cfg is not None or self.no_generation_cfg is not None:
            self.initial_condition(no_generation_cfg=no_generation_cfg)

        # 4. Capture what THIS upcoming episode runs with, so the NEXT reset's
        #    save_data() finalizes it correctly even after generation moves on.
        self._running_mode = self.mode
        self._running_type_mode = self.type_mode
        self._running_csv_init = self.csv_path_init

        # 5. Point PhysiCell's native output at THIS episode's own folder before
        #    the inner reset runs. Otherwise PhysiCell dumps its startup files
        #    (svg/.mat/IC copy) into the PREVIOUS episode's directory — that is
        #    how a train network_field IC leaked into a test episode folder.
        upcoming_dir = self._episode_output_dir(next_episode)
        os.makedirs(upcoming_dir, exist_ok=True)
        self.change_xml(
            keys=["//save/folder", "//save/SVG/enable", "//save/full_data/enable"],
            elements=[upcoming_dir, "false", "false"],
        )

        obs, info = self.env.reset(seed=seed, options=options)

        # fresh action history for the new episode; seed with the midpoint of
        # the action space so the very first step is also delta-constrained
        if self._action_delta_max is not None:
            mid = ((self._action_space.low + self._action_space.high) / 2.0).astype(
                np.float32
            )
            self._action_history = [mid]
            self._decision_anchor = mid.copy()
            self._last_raw_decision = mid.copy()
        else:
            self._action_history = []
            self._decision_anchor = None
            self._last_raw_decision = None
        self._q_calib_buffer = []

        info["train_test"] = self.mode
        info["type_mode"] = self.type_mode
        info["step_episode"] = 0

        return obs, info

    def step(self, action: np.ndarray):
        # hard delta-clip against the DECISION ANCHOR (first action of the current
        # decision block, not the most recent sub-step).  This prevents ratcheting
        # when the same raw action is repeated across multiple sub-steps.
        action = np.asarray(action, dtype=np.float32).copy()
        if self._action_delta_max is not None and self._action_history:
            # Detect a new decision by comparing raw action to the raw action of the
            # current block — not the clipped version, which always differs after clip.
            is_new_decision = not np.allclose(
                action, self._last_raw_decision, atol=1e-5
            )
            if is_new_decision:
                # advance anchor to the last clipped action before this new decision
                self._decision_anchor = self._action_history[-1].copy()
                self._last_raw_decision = action.copy()
            action = np.clip(
                action,
                self._decision_anchor - self._action_delta_max,
                self._decision_anchor + self._action_delta_max,
            )
            action = np.clip(action, self._action_space.low, self._action_space.high)

        d_action = {
            v: np.array([val]) for v, val in zip(self.list_variable_name, action)
        }

        max_radius = np.sqrt(
            (self.env.unwrapped.width / 2) ** 2 + (self.env.unwrapped.height / 2) ** 2
        )

        if self.action_mode == "full":
            cx = self.env.unwrapped.x_min + self.env.unwrapped.width / 2
            cy = self.env.unwrapped.y_min + self.env.unwrapped.height / 2
            d_action["drug_1_x"] = np.array([cx])
            d_action["drug_1_y"] = np.array([cy])
            d_action["drug_1_radius"] = np.array([max_radius])
        else:
            d_action["drug_1_x"] = (
                self.env.unwrapped.x_min
                + d_action["drug_1_x"] * self.env.unwrapped.width
            )
            d_action["drug_1_y"] = (
                self.env.unwrapped.y_min
                + d_action["drug_1_y"] * self.env.unwrapped.height
            )
            radius_min_norm = 0.05  # floor: ~5% of max_radius
            radius_max_norm = 0.20  # ceiling: ~20% of max_radius (precise drops)
            d_action["drug_1_radius"] = (
                radius_min_norm
                + d_action["drug_1_radius"] * (radius_max_norm - radius_min_norm)
            ) * max_radius

        # ── segfault forensics (opt-in via PHYSIGYM_CRASH_LOG=<dir>) ──────────
        # physicell.step() inside self.env.step() can segfault the worker on
        # certain cell states random actions reach. A segfault leaves no Python
        # traceback, so dump the pre-step action + sim state to a per-PID file
        # and fsync it BEFORE stepping. If the step crashes, that file holds the
        # exact action/state that triggered it; on success we truncate it.
        _crash_dir = os.environ.get("PHYSIGYM_CRASH_LOG")
        if _crash_dir:
            try:
                env_u = self.env.unwrapped
                df = env_u.df_alive
                oob = int(
                    (
                        (df["x"] < env_u.x_min)
                        | (df["x"] > env_u.x_max)
                        | (df["y"] < env_u.y_min)
                        | (df["y"] > env_u.y_max)
                    ).sum()
                )
                os.makedirs(_crash_dir, exist_ok=True)
                _cf = os.path.join(_crash_dir, f"laststep_pid{os.getpid()}.txt")
                with open(_cf, "w") as _fh:
                    _fh.write(
                        f"episode={env_u.episode} step={env_u.step_episode}\n"
                        f"n_cells={len(df)} n_out_of_domain={oob}\n"
                        f"raw_action(dose,x,y,r)={action.tolist()}\n"
                        f"applied drug_1_x={float(d_action['drug_1_x'][0])} "
                        f"drug_1_y={float(d_action['drug_1_y'][0])} "
                        f"drug_1_radius={float(d_action['drug_1_radius'][0])} "
                        f"drug_1_dose={float(d_action['drug_1_dose'][0])}\n"
                        f"domain x[{env_u.x_min},{env_u.x_max}] "
                        f"y[{env_u.y_min},{env_u.y_max}]\n"
                    )
                    _fh.flush()
                    os.fsync(_fh.fileno())
            except Exception:
                pass

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)

        if _crash_dir:
            # survived the step → this action was not the culprit; clear the marker
            try:
                open(
                    os.path.join(_crash_dir, f"laststep_pid{os.getpid()}.txt"), "w"
                ).close()
            except Exception:
                pass

        dose_spent = self.env.unwrapped.get_wrapper_attr("get_dose_spent")()

        # ── mechanistic signals for reward analysis ──────────────────
        # T cells (not the drug) do the killing; the drug only modulates the
        # microenvironment. Log per-step effector populations and the raw tumor
        # decrease so reward shaping can be attributed offline.
        env_u = self.env.unwrapped
        try:
            df_alive = env_u.df_alive
            n_tcell = int((df_alive["type"] == "t_cell").sum())
            n_macrophage = int((df_alive["type"] == "macrophage").sum())
        except Exception:
            n_tcell, n_macrophage = 0, 0
        # tumor cells removed this step (>0 means kill); normalised reward is r_cancer_cells
        tumor_killed = float(getattr(env_u, "c_prev", 0) or 0) - float(
            getattr(env_u, "c_t", 0) or 0
        )

        info.update(
            {
                "dose_spent": dose_spent,
                "type_mode": self.type_mode,
                "step_episode": self.env.unwrapped.step_episode,
                "train_test": self.mode,
            }
        )

        # action smoothness penalty: penalise abrupt changes between consecutive actions
        action_arr = np.asarray(action, dtype=np.float32)
        if self._action_history and self.w_smooth > 0.0:
            prev_action = self._action_history[-1]
            smooth_penalty = float(np.sum((action_arr - prev_action) ** 2))
        else:
            smooth_penalty = 0.0

        reward = (
            self.w_cell * r_cancer_cells
            - self.w_dose * dose_spent
            - self.w_smooth * smooth_penalty
        )

        # raw action components in [0, 1] — same values the video annotates
        action_dose = float(action[0]) if len(action) > 0 else 0.0
        action_x = float(action[1]) if len(action) > 1 else 0.0
        action_y = float(action[2]) if len(action) > 2 else 0.0
        action_radius = float(action[3]) if len(action) > 3 else 0.0

        # track raw action vector for episode-end smoothness metrics
        self._action_history.append(action_arr.copy())

        row = {
            "step": self.env.unwrapped.step_episode,
            "reward": reward,
            "r_cancer_cells": float(r_cancer_cells),
            "dose_spent": dose_spent,
            "number_tumor": info.get("number_tumor", 0),
            "n_tcell": n_tcell,
            "n_macrophage": n_macrophage,
            "tumor_killed": tumor_killed,
            "smooth_penalty": smooth_penalty,
            "train_test": self.mode,
            "type_mode": self.type_mode,
            "action_dose": action_dose,
            "action_x": action_x,
            "action_y": action_y,
            "action_radius": action_radius,
        }
        self.list_data.append(row)

        # collect (obs, action, reward) for Q-value calibration on test episodes
        if self.generate_physicell_data:
            self._q_calib_buffer.append(
                {
                    "obs": obs,  # next_obs — what the critic sees as current state
                    "action": action_arr.copy(),
                    "reward": reward,
                }
            )

        # on episode end, attach smoothness metrics and Q-calibration data to info
        if terminated or truncated:
            stats = self._compute_action_smoothness(self._action_history)
            info.update(stats)
            if self.generate_physicell_data and self._q_calib_buffer:
                info["q_calibration_data"] = self._q_calib_buffer.copy()
            self._q_calib_buffer = []
            # reward-analysis hook: surface the finished episode's per-step raw
            # components through info so a parallel SubprocVecEnv can collect them
            # (the subprocess auto-resets after done, wiping self.list_data).
            if getattr(self, "emit_reward_analysis_rows", False):
                info["reward_analysis_rows"] = [dict(r) for r in self.list_data]

        # capture spatial frame for test episodes only
        if self.generate_physicell_data:
            env_inner = self.env.unwrapped
            self._frame_buffer.append(
                {
                    "cells": env_inner.get_matrix_cells().copy(),  # uint8 (n_types, H, W)
                    "subs": env_inner.get_matrix_substrates().copy(),  # uint8 (n_subs,  H, W)
                    "action": action.copy(),  # raw [0,1] values
                    # physical injection geometry as actually applied to the sim
                    # (same transform as d_action), so the overlay circle matches
                    # the drug substrate instead of re-deriving with a different
                    # formula. See _render_frame.
                    "inj_x": float(d_action["drug_1_x"][0]),
                    "inj_y": float(d_action["drug_1_y"][0]),
                    "inj_radius": float(d_action["drug_1_radius"][0]),
                    "reward": reward,
                    "dose": dose_spent,
                    "n_tumor": info.get("number_tumor", 0),
                }
            )

        return obs, reward, terminated, truncated, info

    # ── Action smoothness ────────────────────────────────────────

    @staticmethod
    def _compute_action_smoothness(action_history):
        """
        Per-episode summary of how 'twitchy' the policy was.

        Returns a dict with:
          action_step_delta_mean : mean ||a_t - a_{t-1}||_2 across the episode
          action_step_delta_std  : std of the same series
          action_autocorr_lag1   : mean Pearson correlation between a_t and a_{t-1},
                                   averaged across the 4 action dimensions
                                   (1 = perfectly smooth, 0 = noise, <0 = oscillating)
        Returns NaNs (as 0.0) on episodes too short to compute.
        """
        if len(action_history) < 2:
            return {
                "action_step_delta_mean": 0.0,
                "action_step_delta_std": 0.0,
                "action_autocorr_lag1": 0.0,
            }

        A = np.stack(action_history, axis=0).astype(np.float32)  # (T, D)
        deltas = np.linalg.norm(np.diff(A, axis=0), axis=1)  # (T-1,)

        # per-dimension lag-1 autocorrelation, then average
        ac = []
        for d in range(A.shape[1]):
            x = A[:, d]
            if x.std() < 1e-8:
                continue  # constant series → undefined, skip
            num = np.mean((x[:-1] - x[:-1].mean()) * (x[1:] - x[1:].mean()))
            den = x[:-1].std() * x[1:].std()
            if den > 1e-8:
                ac.append(num / den)

        return {
            "action_step_delta_mean": float(deltas.mean()),
            "action_step_delta_std": float(deltas.std()),
            "action_autocorr_lag1": float(np.mean(ac)) if ac else 0.0,
        }

    # ── Telemetry & video ────────────────────────────────────────

    def save_data(self):
        run_idx = self.env.unwrapped.episode
        if run_idx < 0 or not self.list_data:
            return

        # Finalize using the mode/IC the FINISHED episode actually ran with —
        # captured at its generation time. self.mode / self.csv_path_init have
        # already been advanced for the next episode by this point.
        finished_mode = (
            self._running_mode if self._running_mode is not None else self.mode
        )
        finished_csv = (
            self._running_csv_init
            if self._running_csv_init is not None
            else self.csv_path_init
        )

        out_dir = self._episode_output_dir(run_idx, mode=finished_mode)
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(self.list_data)
        if "dose_spent" in df.columns:
            df["cumulative_dose_spent"] = df["dose_spent"].cumsum()
        if "reward" in df.columns:
            df["cumulative_reward"] = df["reward"].cumsum()

        df.to_csv(os.path.join(out_dir, "data.csv"), index=False)
        shutil.copy(
            finished_csv,
            os.path.join(out_dir, os.path.basename(finished_csv)),
        )

        finished_type_mode = (
            self._running_type_mode
            if self._running_type_mode is not None
            else self.type_mode
        )

        # Dump the raw frame buffer to frames.npz (a cheap numpy write, no
        # rendering). video_maker.py renders it into video.mp4 afterward, so the
        # simulation loop is not slowed by matplotlib/ffmpeg work.
        if self.generate_physicell_data and self._frame_buffer:
            try:
                self._dump_frames(
                    out_dir=out_dir,
                    run_idx=run_idx,
                    type_mode=finished_type_mode,
                    frame_buffer=self._frame_buffer,
                )
            except Exception as e:
                print(
                    f"[PhysiCellModelWrapper] frame dump failed for run {run_idx}: {e}"
                )

        self.list_data = []
        self._frame_buffer = []

        # cleanup: keep video.mp4, frames.npz (rendered later by video_maker.py),
        # and the FINISHED episode's own IC. Any other CSV here (e.g. the next
        # episode's IC copied by PhysiCell) would be a leak from a different
        # mode — drop it.
        keep_csv = os.path.basename(finished_csv)
        for f in os.scandir(out_dir):
            if not f.is_file():
                continue
            if f.name == "frames.npz":
                continue
            ext = os.path.splitext(f.name)[1]
            if ext == ".mp4":
                continue
            if ext == ".csv" and f.name in {keep_csv, "data.csv"}:
                continue
            os.unlink(f.path)

    def _dump_frames(
        self,
        out_dir: str,
        run_idx: int,
        type_mode: str = None,
        frame_buffer: list = None,
    ):
        """Serialize the raw frame buffer + episode metadata to out_dir/frames.npz.

        This intentionally does NO rendering: no matplotlib, no cv2, no ffmpeg —
        only a numpy array write. All the expensive per-step figure rendering is
        deferred to video_maker.py, which loads frames.npz and builds video.mp4
        as a post-processing step, so it never slows down the simulation.

        Args:
            out_dir: directory that receives frames.npz
            run_idx: episode index
            type_mode: geometry type (network_field, circular, etc)
            frame_buffer: list of frame dicts. If None, uses self._frame_buffer (for backward compat).
        """
        if type_mode is None:
            type_mode = self.type_mode
        if frame_buffer is None:
            frame_buffer = self._frame_buffer

        if not frame_buffer:
            return

        env_inner = self.env.unwrapped
        cell_type_names = list(env_inner.cell_type_to_id.keys())
        cell_type_colors = env_inner.cell_type_to_color
        substrate_names = list(env_inner.substrate_unique)

        # Stack per-step arrays; keep everything numpy so np.savez is the only cost.
        cells = np.stack(
            [f["cells"] for f in frame_buffer], axis=0
        )  # (T, n_types, H, W)
        subs = np.stack([f["subs"] for f in frame_buffer], axis=0)  # (T, n_subs,  H, W)
        actions = np.stack([np.asarray(f["action"]) for f in frame_buffer], axis=0)
        rewards = np.asarray([f["reward"] for f in frame_buffer], dtype=np.float32)
        doses = np.asarray([f["dose"] for f in frame_buffer], dtype=np.float32)
        # physical injection geometry actually applied to the sim (see append site)
        inj_x = np.asarray(
            [f.get("inj_x", np.nan) for f in frame_buffer], dtype=np.float32
        )
        inj_y = np.asarray(
            [f.get("inj_y", np.nan) for f in frame_buffer], dtype=np.float32
        )
        inj_r = np.asarray(
            [f.get("inj_radius", np.nan) for f in frame_buffer], dtype=np.float32
        )

        np.savez_compressed(
            os.path.join(out_dir, "frames.npz"),
            cells=cells,
            subs=subs,
            actions=actions,
            rewards=rewards,
            doses=doses,
            inj_x=inj_x,
            inj_y=inj_y,
            inj_radius=inj_r,
            # metadata needed by _render_frame, stored as object arrays / scalars
            cell_type_names=np.array(cell_type_names, dtype=object),
            cell_type_colors=np.array(
                [cell_type_colors[n] for n in cell_type_names], dtype=object
            ),
            substrate_names=np.array(substrate_names, dtype=object),
            type_mode=str(type_mode),
            episode=int(run_idx),
            x_min=float(env_inner.x_min),
            x_max=float(env_inner.x_max),
            y_min=float(env_inner.y_min),
            y_max=float(env_inner.y_max),
        )

    def _episode_output_dir(self, run_idx: int, mode: str = None) -> str:
        return os.path.join(
            self.base_output_dir,
            mode if mode is not None else self.mode,
            "episodes",
            f"run_{str(run_idx).zfill(6)}",
        )
