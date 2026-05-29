from pathlib import Path
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


data_dir = Path("data")
wandb_dir = Path("wandb_csv_exports")


def copy_item(args):
    src, dst = args

    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def collect_tasks():
    tasks = []

    for data_subdir in data_dir.iterdir():
        if not data_subdir.is_dir():
            continue

        target_subdir = wandb_dir / data_subdir.name
        if not target_subdir.exists():
            continue

        for item in data_subdir.iterdir():
            tasks.append((item, target_subdir / item.name))

    return tasks


if __name__ == "__main__":
    tasks = collect_tasks()

    print(f"Copying {len(tasks)} items using {cpu_count()} processes")

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(copy_item, tasks), total=len(tasks)))
