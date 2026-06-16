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
    idx = np.random.choice(len(coords), size=n, p=probs, replace=True)

    return coords[idx]


def df_cells(x, y, cell_type):
    return pd.DataFrame({"x": x, "y": y, "z": 0, "type": cell_type})


# ============================================================
# Population Generators
# ============================================================


def circular_mode(params, bounds):
    nb_tumor_cells = params["tumor"]["number_cells"]
    nb_cell_1 = params["macrophage"]["number_cells"]
    nb_t_cell = params["t_cell"]["number_cells"]

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Calculate half-width and half-height for scaling radii
    hw, hh = (x_max - x_min) / 2, (y_max - y_min) / 2

    # 1. Define the exact center of the bounding box for the Tumor
    center_y = y_min + 0.5 * (y_max - y_min)
    tumor_center = (x_min + 0.5 * (x_max - x_min), center_y)

    # 2. Define the left (20%) and right (80%) centers for the immune cells
    immune_centers = [
        (x_min + 0.2 * (x_max - x_min), center_y),
        (x_min + 0.8 * (x_max - x_min), center_y),
    ]

    # Shuffle only the immune centers so Macrophage and T_cells randomly swap left/right
    random.shuffle(immune_centers)

    # Tumor cluster (Always uses tumor_center)
    tx, ty = ellipse_points(
        nb_tumor_cells,
        params["r1"] * hw,
        params["r2_t"] * hh,
        tumor_center,
        jitter=params["jit_t"],
    )

    # Macrophage cluster (Uses one of the shuffled immune centers)
    cx1, cy1 = ellipse_points(
        nb_cell_1,
        params["r1_c"] * hw,
        params["r2_c"] * hh,
        immune_centers[0],
        jitter=params["jit_c"],
    )

    # T_cell cluster (Uses the other shuffled immune center)
    dx1, dy1 = ellipse_points(
        nb_t_cell,
        params["r1_c"] * hw,
        params["r2_c"] * hh,
        immune_centers[1],
        jitter=params["jit_c"],
    )

    # Build DataFrames and concatenate
    tumor = df_cells(tx, ty, "tumor")
    macrophage = df_cells(cx1, cy1, "macrophage")
    t_cell = df_cells(dx1, dy1, "t_cell")

    return pd.concat([tumor, macrophage, t_cell], ignore_index=True)


def random_mode(params, bounds):
    nb_tumor_cells = params["tumor"]["number_cells"]
    nb_cell_1 = params["macrophage"]["number_cells"]
    nb_t_cell = params["t_cell"]["number_cells"]

    tumor = df_cells(
        np.random.uniform(bounds[0][0], bounds[0][1], nb_tumor_cells),
        np.random.uniform(bounds[1][0], bounds[1][1], nb_tumor_cells),
        "tumor",
    )

    macrophage = df_cells(
        np.random.uniform(bounds[0][0], bounds[0][1], nb_cell_1),
        np.random.uniform(bounds[1][0], bounds[1][1], nb_cell_1),
        "macrophage",
    )

    t_cell = df_cells(
        np.random.uniform(bounds[0][0], bounds[0][1], nb_t_cell),
        np.random.uniform(bounds[1][0], bounds[1][1], nb_t_cell),
        "t_cell",
    )

    return pd.concat([tumor, macrophage, t_cell])


def rectangle_mode(params, bounds):
    nb_tumor_cells = params["tumor"]["number_cells"]
    nb_macrophage = params["macrophage"]["number_cells"]
    nb_t_cell = params["t_cell"]["number_cells"]

    # Generate 3 base positions (fractions of x-axis)
    values = [
        random.uniform(0.1, 0.2),
        random.uniform(0.3, 0.5),
        random.uniform(0.6, 0.8),
    ]

    # Shuffle them (in-place)
    random.shuffle(values)

    width = 0.1  # width of each rectangle (fraction of x-axis)
    x_min_bound, x_max_bound = bounds[0]
    y_min_bound, y_max_bound = bounds[1]
    x_range = x_max_bound - x_min_bound

    def make_x_interval(v):
        x_min = x_min_bound + v * x_range
        x_max = x_min + width * x_range
        return x_min, x_max

    # Tumor
    x_min, x_max = make_x_interval(values[0])
    tumor = df_cells(
        np.random.uniform(x_min, x_max, nb_tumor_cells),
        np.random.uniform(y_min_bound, y_max_bound, nb_tumor_cells),
        "tumor",
    )

    # Macrophage
    x_min, x_max = make_x_interval(values[1])
    macrophage = df_cells(
        np.random.uniform(x_min, x_max, nb_macrophage),
        np.random.uniform(y_min_bound, y_max_bound, nb_macrophage),
        "macrophage",
    )

    # T cells
    x_min, x_max = make_x_interval(values[2])
    t_cell = df_cells(
        np.random.uniform(x_min, x_max, nb_t_cell),
        np.random.uniform(y_min_bound, y_max_bound, nb_t_cell),
        "t_cell",
    )

    return pd.concat([tumor, macrophage, t_cell], ignore_index=True)


def generate_synthetic_network_field(params, bounds, amplitude=1, save=False):
    domain_size = int(bounds[0][1] - bounds[0][0]), int(bounds[1][1] - bounds[1][0])
    fields = generate_balanced_fields(
        domain_size=domain_size, params=params, amplitude=amplitude
    )

    xs_final, ys_final, phenotypes_final = [], [], []
    n_types = len(list(params.keys()))
    fig, axes = (
        plt.subplots(n_types, 2, figsize=(10, 5 * n_types)) if save else None,
        None,
    )
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
            ax_field = axes[row_idx, 0]
            im = ax_field.imshow(field, cmap="viridis")
            ax_field.set_title(f"Field: {ct}")
            ax_field.axis("off")
            fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04)
            ax_scatter = axes[row_idx, 1]
            ax_scatter.scatter(ys, domain_size[1] - xs, s=10, alpha=0.8)
            ax_scatter.set_title(f"Cells: {ct}")
            ax_scatter.set_xlabel("X")
            ax_scatter.set_ylabel("Y")
            ax_scatter.set_aspect("equal")

        xs_final.extend(xs)
        ys_final.extend(ys)
        phenotypes_final.extend([ct] * len(coords))

    df_result = pd.DataFrame(
        {
            "x": xs_final,
            "y": ys_final,
            "z": [0] * len(xs_final),
            "type": phenotypes_final,
        }
    )
    # Shift by bounds origin — cast to int to preserve integer dtype
    df_result["x"] = (df_result["x"] + int(bounds[0][0])).astype(int)
    df_result["y"] = (df_result["y"] + int(bounds[1][0])).astype(int)
    return df_result


# ============================================================
# CSV + Plot
# ============================================================
def generate_initial_condition(
    csv_path, mode, x_min, x_max, y_min, y_max, params, seed=42
):
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

    # =========================================================================
    # NEW LOGIC: Round to nearest whole number, cast to integer, drop overlaps
    # =========================================================================
    df["x"] = df["x"].round().astype(int).clip(x_min, x_max)
    df["y"] = df["y"].round().astype(int).clip(y_min, y_max)

    # Changed keep=False to keep='first' so that one cell remains if multiples
    # land on the exact same (x, y) grid spot.
    df = df.drop_duplicates(subset=["x", "y"], keep="first")

    # No longer using float_format="%.6f" since they are pure integers
    df.to_csv(csv_path, index=False)
    return df, mode


def plot_cells(df, path):
    colors = {"tumor": "grey", "macrophage": "blue", "t_cell": "green"}
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
    x_min, x_max, y_min, y_max = 0, 63, 0, 63

    modes = ["network_field"]
    params = {
        "tumor": {"correlation_length": 35, "threshold": 0.55, "number_cells": 512},
        "macrophage": {
            "correlation_length": 35,
            "threshold": 0.55,
            "number_cells": 128,
        },
        "t_cell": {"correlation_length": 35, "threshold": 0.55, "number_cells": 42},
    }

    d_arg_generation = {
        "csv_path": None,
        "params": params,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "seed": None,
        "mode": None,
    }
    seed = 42
    for i in range(1, 48):
        # Fixed csv_path extension from .png to .csv
        d_arg_generation["csv_path"] = f"{out}/cells_{i}.csv"
        d_arg_generation["seed"] = seed
        d_arg_generation["mode"] = "network_field"
        d_arg_generation["params"] = {
            "tumor": {
                "correlation_length": 35,
                "threshold": 0.02 * i,
                "number_cells": 512,
            },
            "macrophage": {
                "correlation_length": 35,
                "threshold": 0.02 * i,
                "number_cells": 128,
            },
            "t_cell": {
                "correlation_length": 35,
                "threshold": 0.02 * i,
                "number_cells": 42,
            },
        }
        df, type_mode = generate_initial_condition(**d_arg_generation)
        plot_cells(df, f"{out}/cells_{i * 0.02}.png")
