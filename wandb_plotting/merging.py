import os
import re
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use("tableau-colorblind10")


def parse_folder_name(folder_name):
    """
    Returns (seed:int, obs_type:str) or (None, None) if parsing fails
    """
    pattern = r"async_sac_tip_(\d+)_([a-zA-Z0-9_]+)_\d+"
    match = re.match(pattern, folder_name)
    if not match:
        return None, None
    seed = int(match.group(1))
    obs_type = match.group(2)
    return seed, obs_type


csv_root = "./wandb_csv_exports"


def time_weighted_ema(series, steps, tau):
    """
    series: pd.Series of values
    steps: pd.Series or array of step indices
    tau: smoothing timescale (in steps)
    """
    ema = np.zeros(len(series))
    ema[0] = series.iloc[0]

    for i in range(1, len(series)):
        dt = steps.iloc[i] - steps.iloc[i - 1]
        alpha = 1.0 - np.exp(-dt / tau)
        ema[i] = alpha * series.iloc[i] + (1 - alpha) * ema[i - 1]

    return pd.Series(ema, index=series.index)


def load_all_runs(csv_root):
    """
    Returns:
        dict[obs_type] -> list of DataFrames (one per seed)
    """
    groups = defaultdict(list)

    for folder in os.listdir(csv_root):
        folder_path = os.path.join(csv_root, folder)
        if not os.path.isdir(folder_path):
            continue

        seed, obs_type = parse_folder_name(folder)
        if seed is None:
            continue

        csv_path = os.path.join(folder_path, "history.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df = df[["step", "return"]].copy()
        df["seed"] = seed

        groups[obs_type].append(df)

    return groups


groups = load_all_runs(csv_root)


def prepare_group_data_ema_then_mean(all_data, alpha=0.1):
    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)

    reshaped = df.pivot_table(
        index="step",
        columns="seed",
        values="return",
        aggfunc="mean",
    ).sort_index()

    # EWMA per seed (ignores step spacing, as requested)
    smoothed = reshaped.copy()
    for seed in smoothed.columns:
        smoothed[seed] = smoothed[seed].ewm(alpha=alpha, adjust=False).mean()

    # Aggregate across seeds
    out = pd.DataFrame(index=smoothed.index)
    out["mean"] = smoothed.mean(axis=1)
    out["std"] = smoothed.std(axis=1)

    return out


def prepare_group_data_rolling_then_mean(all_data, window=100):
    """
    all_data: list of DataFrames with columns [step, return, seed]
    """
    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    # Pivot: step × seed
    reshaped = df.pivot(index="step", columns="seed", values="return")
    reshaped = reshaped.sort_index()
    reshaped = reshaped.interpolate(method="linear", axis=0)

    # Rolling mean per seed
    smoothed = reshaped.copy()
    for seed in smoothed.columns:
        smoothed[seed] = (
            smoothed[seed].rolling(window=window, center=True, min_periods=1).mean()
        )

    # Aggregate across seeds
    out = pd.DataFrame(index=smoothed.index)
    out["mean"] = smoothed.mean(axis=1)
    out["min"] = smoothed.min(axis=1)
    out["max"] = smoothed.max(axis=1)
    out["std"] = smoothed.std(axis=1)

    return out


def prepare_aligned_returns(all_data):
    """
    all_data: list of DataFrames [step, return, seed]
    Returns:
        steps: np.array shape (T,)
        returns: np.array shape (num_seeds, T)
    """
    df = pd.concat(all_data, ignore_index=True)

    reshaped = df.pivot_table(
        index="step",
        columns="seed",
        values="return",
        aggfunc="mean",
    )

    steps = reshaped.index.to_numpy()
    returns = reshaped.to_numpy().T  # (num_seeds, T)
    return steps, returns


def bootstrap_mean_ewma(all_data, alpha=0.01, B=3000):
    """
    Bootstrap CI over seeds using Pandas EWMA.
    """
    df = pd.concat(all_data, ignore_index=True)

    reshaped = (
        df.pivot_table(
            index="step",
            columns="seed",
            values="return",
            aggfunc="mean",
        )
        .sort_index()
        .interpolate(method="linear", axis=0)
    )

    steps = reshaped.index
    returns = reshaped.to_numpy().T  # (num_seeds, T)
    num_seeds, T = returns.shape

    bootstrap_curves = np.zeros((B, T))

    for b in range(B):
        idx = np.random.choice(num_seeds, size=num_seeds, replace=True)
        mean_curve = returns[idx].mean(axis=0)

        bootstrap_curves[b] = (
            pd.Series(mean_curve, index=steps)
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy()
        )

    # Central estimate
    mean_curve = returns.mean(axis=0)
    mean_ewma = (
        pd.Series(mean_curve, index=steps)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .to_numpy()
    )

    out = pd.DataFrame(index=steps)
    out["mean"] = mean_ewma
    out["ci_low"] = np.percentile(bootstrap_curves, 2.5, axis=0)
    out["ci_high"] = np.percentile(bootstrap_curves, 97.5, axis=0)

    return out


def save_plot_aggregation(
    bootstrap: bool = False,
    alpha=0.01,
    B: int = 256,
    figure_name: str = "performance.pdf",
):
    aggregated = {}

    for obs_type, dfs in groups.items():
        if bootstrap:
            aggregated[obs_type] = bootstrap_mean_ewma(
                dfs,
                alpha=alpha,  # same scale as before
                B=B,  # 3k–5k is fine
            )
        else:
            aggregated[obs_type] = prepare_group_data_ema_then_mean(dfs, alpha=alpha)

    fig, (ax_main, ax_zoom) = plt.subplots(
        ncols=2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 1]},  # 75% / 25%
    )

    # ---- main plot (full range) ----
    for obs_type, df in aggregated.items():
        if df is None:
            continue
        ax_main.plot(df.index, df["mean"], label=obs_type)
        if bootstrap:
            ax_main.fill_between(
                df.index,
                df["ci_low"],
                df["ci_high"],
                alpha=0.2,
            )
            print(
                obs_type,
                round(df["mean"].iloc[-1], 2),
                round(df["ci_low"].iloc[-1], 2),
                round(df["ci_high"].iloc[-1], 2),
            )
        else:
            ax_main.fill_between(
                df.index,
                df["mean"] - df["std"],
                df["mean"] + df["std"],
                alpha=0.2,
            )
            print(
                obs_type,
                round(df["mean"].iloc[-1], 2),
                round(df["std"].iloc[-1], 2),
            )

    ax_main.set_xlabel("Step")
    ax_main.set_ylabel("Return")
    ax_main.set_title("Performance for different state spaces")
    ax_main.legend()
    ax_main.grid(True)

    # ---- zoom plot (restricted range) ----
    for obs_type, df in aggregated.items():
        if df is None:
            continue

        ax_zoom.plot(df.index, df["mean"])
        if bootstrap:
            ax_zoom.fill_between(
                df.index,
                df["ci_low"],
                df["ci_high"],
                alpha=0.2,
            )
        else:
            ax_zoom.fill_between(
                df.index,
                df["mean"] - df["std"],
                df["mean"] + df["std"],
                alpha=0.2,
            )

    # Zoom limits
    ax_zoom.set_xlim(1.75e5, 2e5)
    ax_zoom.set_ylim(160, 185)

    ax_zoom.set_title("Zoom")
    ax_zoom.grid(True)

    for ax in [ax_main, ax_zoom]:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        # Optional: Ensure the offset text (the 'x10^5') is visible and formatted nicely
        ax.xaxis.get_offset_text().set_fontsize(10)

    fig.savefig(fname=figure_name, format="pdf", bbox_inches="tight")

    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    # save_plot_aggregation(figure_name="no_bootstrap_performance.pdf", bootstrap=False)
    save_plot_aggregation(figure_name="bootstrap_performance.pdf", bootstrap=True)
