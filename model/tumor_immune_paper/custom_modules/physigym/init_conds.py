import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from math import sqrt
from scipy.ndimage import gaussian_filter
# ============================================================
# Utils
# ============================================================


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def ellipse_points(n, r1, r2, center, angle=0.0, jitter=0.0, perimeter=False):
    a = (
        np.linspace(0, 2 * np.pi, n, endpoint=False)
        if perimeter
        else np.random.uniform(0, 2 * np.pi, n)
    )
    r = np.ones(n) if perimeter else np.sqrt(np.random.rand(n))
    x, y = r * r1 * np.cos(a), r * r2 * np.sin(a)

    if angle != 0:
        c, s = np.cos(angle), np.sin(angle)
        x, y = c * x - s * y, s * x + c * y

    x += center[0] + np.random.normal(0, jitter, n)
    y += center[1] + np.random.normal(0, jitter, n)
    return x, y


def generate_correlated_field(
    domain_size,
    correlation_length,
):
    noise = np.random.randn(*domain_size)
    sigma = correlation_length / sqrt(2)
    field = gaussian_filter(noise, sigma=sigma, mode="reflect")
    return field


def generate_balanced_fields(
    domain_size,
    params,
    amplitude=1.0,
    local_noise_level=0.3,
):
    local_noise_level = local_noise_level / 100 + 0.05

    fields = {}
    for key in params.keys():
        correlation_length = params[key]["correlation_length"]
        # === Génère un bruit local filtré ===
        local_noise = np.random.randn(*domain_size)
        filtered_noise = gaussian_filter(local_noise, sigma=correlation_length / 3)

        # === Champ final : base + bruit doux ===
        final_field = amplitude * (
            generate_correlated_field(domain_size, correlation_length)
            + local_noise_level * filtered_noise
        )
        min_final_field = np.min(final_field)
        max_final_field = np.max(final_field)
        fields[key] = (max_final_field - final_field) / (
            max_final_field - min_final_field
        )

    return fields


def weighted_pick(arr, threshold, n=1):
    # mask only the values above threshold
    mask = arr > threshold

    # get coordinates of valid pixels
    coords = np.argwhere(mask)

    # get the corresponding probabilities
    probs = arr[mask].astype(float)

    # normalize probabilities to sum to 1
    probs /= probs.sum()

    # weighted choice
    idx = np.random.choice(len(coords), size=n, p=probs, replace=False)

    return coords[idx]


def df_cells(x, y, cell_type):
    return pd.DataFrame({"x": x, "y": y, "z": 0.0, "type": cell_type})


# ============================================================
# Population Generators
# ============================================================


def circular_mode(params, bounds):
    cx, cy = np.mean(bounds[0]), np.mean(bounds[1])
    hw, hh = (bounds[0][1] - bounds[0][0]) / 2, (bounds[1][1] - bounds[1][0]) / 2

    tx, ty = ellipse_points(
        params["tumor"]["number_cells"],
        params["r1"] * hw,
        params["r2_t"] * hh,
        (cx, cy),
        jitter=params["jit_t"],
    )

    cx1, cy1 = ellipse_points(
        params["cell_1"]["number_cells"],
        params["r1_c"] * hw,
        params["r2_c"] * hh,
        (cx, cy),
        jitter=params["jit_c"],
        perimeter=True,
    )

    return pd.concat([df_cells(tx, ty, "tumor"), df_cells(cx1, cy1, "cell_1")])


def random_mode(params, bounds):
    nb_tumor_cells = params["tumor"]["number_cells"]
    nb_cell_1 = params["cell_1"]["number_cells"]

    tumor = df_cells(
        np.random.uniform(
            bounds[0][0],
            bounds[0][1],
            nb_tumor_cells,
        ),
        np.random.uniform(bounds[1][0], bounds[1][1], nb_tumor_cells),
        "tumor",
    )

    cell1 = df_cells(
        np.random.uniform(bounds[0][0], bounds[0][1], nb_cell_1),
        np.random.uniform(bounds[1][0], bounds[1][1], nb_cell_1),
        "cell_1",
    )

    return pd.concat([tumor, cell1])


def rectangle_mode(params, bounds):
    nb_tumor_cells = params["tumor"]["number_cells"]
    nb_cell_1 = params["cell_1"]["number_cells"]
    value_tumor = random.uniform(0.2, 0.4)
    value_cell_1 = random.uniform(0.7, 0.9)

    tumor = df_cells(
        np.random.uniform(
            bounds[0][0],
            bounds[0][0] + value_tumor * (bounds[0][1] - bounds[0][0]),
            nb_tumor_cells,
        ),
        np.random.uniform(bounds[1][0], bounds[1][1], nb_tumor_cells),
        "tumor",
    )

    cell1 = df_cells(
        np.random.uniform(bounds[0][1] * value_cell_1, bounds[0][1], nb_cell_1),
        np.random.uniform(bounds[1][0], bounds[1][1], nb_cell_1),
        "cell_1",
    )

    return pd.concat([tumor, cell1])


def generate_synthetic_network_field(
    params,
    bounds,
    amplitude=1,
    save=False,
):
    domain_size = int(bounds[0][1] - bounds[0][0]), int(bounds[1][1] - bounds[1][0])
    # === Generate Fields ===
    fields = generate_balanced_fields(
        domain_size=domain_size,
        params=params,
        amplitude=amplitude,
    )
    xs_final = []
    ys_final = []
    phenotypes_final = []
    n_types = len(list(params.keys()))
    fig, axes = (
        plt.subplots(n_types, 2, figsize=(10, 5 * n_types)) if save else None,
        None,
    )

    # Handle case of single row
    if n_types == 1:
        axes = np.array([axes])

    for row_idx, ct in enumerate(list(params.keys())):
        field = fields[ct]
        coords = weighted_pick(
            field, threshold=params[ct]["threshold"], n=params[ct]["number_cells"]
        )
        xs = coords[:, 0]
        ys = coords[:, 1]
        if save:
            # ========== LEFT: FIELD ==========
            ax_field = axes[row_idx, 0]
            im = ax_field.imshow(field, cmap="viridis")
            ax_field.set_title(f"Field: {ct}")
            ax_field.axis("off")
            fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04)

            # ========== RIGHT: SCATTER CELLS ==========
            ax_scatter = axes[row_idx, 1]
            ax_scatter.scatter(ys, domain_size[1] - xs, s=10, alpha=0.8)
            ax_scatter.set_title(f"Cells: {ct}")
            ax_scatter.set_xlabel("X")
            ax_scatter.set_ylabel("Y")
            ax_scatter.set_aspect("equal")

        xs_final.extend(xs)  # extend, not append
        ys_final.extend(ys)
        phenotypes_final.extend([ct] * len(coords))  # repeat ct for each cell

    df_cells = pd.DataFrame(
        data={
            "x": xs_final,
            "y": ys_final,
            "z": [0] * len(xs_final),
            "type": phenotypes_final,
        }
    )
    df_cells[["x", "y"]] += bounds[0][0], bounds[1][0]
    return df_cells


# ============================================================
# CSV + Plot
# ============================================================
def generate_initial_condition(
    csv_path, mode, x_min, x_max, y_min, y_max, cell_2_fraction, params, seed=42
):
    if cell_2_fraction is None:
        cell_2_fraction = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    cell_2_fraction = (
        np.random.choice(cell_2_fraction)
        if isinstance(cell_2_fraction, (list, np.ndarray))
        else cell_2_fraction
    )
    set_seed(seed)
    bounds = ((x_min, x_max), (y_min, y_max))
    if isinstance(mode, (list, tuple)):
        mode = random.choice(mode)
    if mode == "circular":
        params_1 = dict(
            r1=random.uniform(0.1, 0.4),
            r2_t=random.uniform(0.1, 0.4),
            r1_c=random.uniform(0.5, 1.2),
            r2_c=random.uniform(0.2, 0.6),
            jit_t=random.randint(5, 15),
            jit_c=random.randint(5, 10),
        )
        params_1 |= params
        df = circular_mode(params_1, bounds)

    elif mode == "rectangle":
        df = rectangle_mode(params, bounds)

    elif mode == "random":
        df = random_mode(params, bounds)

    elif mode == "network_field":
        df = generate_synthetic_network_field(params, bounds)

    else:
        raise ValueError(mode)

    cell1_pos = np.flatnonzero(df["type"].values == "cell_1")
    df.iloc[
        np.random.choice(
            cell1_pos, int(cell_2_fraction * len(cell1_pos)), replace=False
        ),
        df.columns.get_loc("type"),
    ] = "cell_2"
    df["x"] = df["x"].clip(x_min, x_max)
    df["y"] = df["y"].clip(y_min, y_max)
    df = df.drop_duplicates(subset=["x", "y"], keep=False)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    return df, mode


def plot_cells(df, path):
    colors = {"tumor": "green", "cell_1": "blue", "cell_2": "red"}
    plt.figure(figsize=(6, 6))
    for t, c in colors.items():
        s = df[df.type == t]
        if len(s):
            plt.scatter(s.x, s.y, s=15, c=c, label=t)
    plt.axis("equal")
    plt.legend()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    out = "configs_network_field"
    os.makedirs(out, exist_ok=True)
    x_min, x_max, y_min, y_max = -256, 256, -256, 256

    modes = ["network_field"]
    params = {
        "tumor": {"correlation_length": 35, "threshold": 0.55, "number_cells": 512},
        "cell_1": {"correlation_length": 35, "threshold": 0.55, "number_cells": 128},
    }

    d_arg_generation = {
        "csv_path": None,
        "params": params,
        "cell_2_fraction": None,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "seed": None,
        "mode": None,
    }
    seed = 42
    for i in range(1, 48):
        d_arg_generation["csv_path"] = f"{out}/cells_{i}.png"
        d_arg_generation["seed"] = seed
        d_arg_generation["mode"] = "network_field"
        d_arg_generation["params"] = {
            "tumor": {
                "correlation_length": 35,
                "threshold": 0.02 * i,
                "number_cells": 512,
            },
            "cell_1": {
                "correlation_length": 35,
                "threshold": 0.02 * i,
                "number_cells": 128,
            },
        }
        df, type_mode = generate_initial_condition(**d_arg_generation)
        plot_cells(df, f"{out}/cells_{i * 0.02}.png")
