import re
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError
import matplotlib


ROOT = "./data"


def get_folder_name(path):
    pattern = r"^(.*)/env\d+/"
    match = re.match(pattern, path)
    return match.group(1)


def parse_folder_name(folder_name):
    pattern = r"^(.+?)_(\d+)_([a-zA-Z0-9_]+)_(\d+)$"
    match = re.match(pattern, folder_name)
    if not match:
        return None, None, None

    experiment_name = match.group(1)
    seed = int(match.group(2))
    state_space = match.group(3)

    return experiment_name, seed, state_space


if __name__ == "__main__":
    tasks = []

    for run_folder in os.listdir(ROOT):
        run_path = os.path.join(ROOT, run_folder)
        if not os.path.isdir(run_path):
            continue

        experiment_name, seed, state_space = parse_folder_name(run_folder)
        if seed is None:
            continue

        for env_name in os.listdir(run_path):
            if not env_name.startswith("env"):
                continue

            env_id = int(env_name.replace("env", ""))
            env_path = os.path.join(run_path, env_name)

            for episode_name in os.listdir(env_path):
                if not episode_name.startswith("episode"):
                    continue

                episode_id = int(episode_name.replace("episode", ""))
                csv_path = os.path.join(env_path, episode_name, "data.csv")

                if not os.path.exists(csv_path):
                    continue

                tasks.append((seed, state_space, env_id, episode_id, csv_path))

    dictionary = defaultdict(list)

    for seed, state_space, env_id, episode_id, csv_path in tasks:
        key = f"{seed}_{env_id}_{episode_id}"
        dictionary[key].append(csv_path)

    filtered_dictionary = {}

    for key, paths in dictionary.items():
        if len(paths) >= 4 and "-1" not in key:
            filtered_dictionary[key] = paths

    dictionary = filtered_dictionary

    METRICS = [
        "reward",
        "mean_drugs",
        "number_tumor",
        "number_cell_1",
        "number_cell_2",
        "cumulative_drugs",
        "cumulative_reward",
    ]

    def plot_single_metric_single_state(key, state_space, df, metric, out_path):
        if metric not in df.columns:
            return

        plt.figure(figsize=(8, 4))
        plt.plot(df["step"], df[metric], alpha=0.9)

        plt.xlabel("step")
        plt.ylabel(metric)
        plt.title(f"{key} — {state_space} — {metric}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs(out_path, exist_ok=True)
        plt.savefig(os.path.join(out_path, f"{metric}.pdf"))
        plt.close()

    def plot_all_metrics_for_key(key, state_space_dfs, metrics, out_path):
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)

        if n_metrics == 1:
            axes = [axes]

        plotted_anything = False

        for i, metric in enumerate(metrics):
            ax = axes[i]

            for state_space, df in state_space_dfs.items():
                if metric not in df.columns:
                    continue

                ax.plot(df["step"], df[metric], label=state_space, alpha=0.85)
                plotted_anything = True

            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(fontsize=9)

        if not plotted_anything:
            plt.close()
            return

        axes[-1].set_xlabel("step")
        fig.suptitle(f"Key {key}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        os.makedirs(out_path, exist_ok=True)
        plt.savefig(os.path.join(out_path, f"{key}_all_metrics.pdf"))
        plt.close()

    def plot_all_state_spaces_for_key(key, state_space_dfs, metrics, key_path):
        for state_space, df in state_space_dfs.items():
            state_path = os.path.join(key_path, state_space)
            os.makedirs(state_path, exist_ok=True)

            for metric in metrics:
                plot_single_metric_single_state(
                    key=key,
                    state_space=state_space,
                    df=df,
                    metric=metric,
                    out_path=state_path,
                )

    def load_key_data(csv_paths):
        data = {}

        for csv_path in csv_paths:
            _, _, state_space = parse_folder_name(get_folder_name(csv_path))

            try:
                df = pd.read_csv(csv_path)

                if df.empty or "step" not in df.columns:
                    continue

                data[state_space] = df

            except (EmptyDataError, FileNotFoundError):
                pass
            except Exception:
                pass

        return data

    def process_key_all_metrics(args):
        key, csv_paths, metrics, base_path = args

        state_space_dfs = load_key_data(csv_paths)

        if not state_space_dfs:
            return None

        key_path = os.path.join(base_path, key)

        plot_all_metrics_for_key(
            key=key, state_space_dfs=state_space_dfs, metrics=metrics, out_path=key_path
        )

        return key

    def process_key(args):
        key, csv_paths, metrics, base_path = args

        state_space_dfs = load_key_data(csv_paths)
        if not state_space_dfs:
            return None

        key_path = os.path.join(base_path, key)
        os.makedirs(key_path, exist_ok=True)

        plot_all_state_spaces_for_key(
            key=key, state_space_dfs=state_space_dfs, metrics=metrics, key_path=key_path
        )

        return key

    matplotlib.use("Agg")  # IMPORTANT for multiprocessing + headless

    path = os.path.join(ROOT, "merge")
    os.makedirs(path, exist_ok=True)

    tasks = [(key, dictionary[key], METRICS, path) for key in dictionary.keys()]

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_key, tasks)):
            pass
