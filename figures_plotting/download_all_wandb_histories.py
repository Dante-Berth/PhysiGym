import wandb
import pandas as pd
import os
import multiprocessing as mp


def process_single_run(args):
    (
        entity,
        project,
        run_id,
        run_name,
        output_dir,
        name_return,
        name_length,
    ) = args

    api = wandb.Api()  # MUST be created inside the process

    run_name = run_name or run_id
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"▶ Processing {run_name} ({run_id})")
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        out_path = os.path.join(run_dir, "history.csv")
        history_iter = run.scan_history(keys=["_step", name_return, name_length])

        records = []
        for row in history_iter:
            records.append(
                {
                    "step": row.get("_step"),
                    "return": row.get(name_return),
                    "length": row.get(name_length),
                }
            )

        df = pd.DataFrame(records)

        if df.empty:
            print(f"⚠️  {run_name}: no valid data")
            return
        
        df.to_csv(out_path, index=False)

        print(f"✅ Saved {run_name}/history.csv")

    except Exception as e:
        print(f"❌ Error {run_name} ({run_id}): {e}")


def download_all_wandb_histories_mp(
    entity: str,
    project: str,
    output_dir: str = "wandb_csv_exports",
    name_return: str = "charts/return",
    name_length: str = "charts/length",
    num_workers: int | None = None,
):
    wandb.login()
    api = wandb.Api()

    os.makedirs(output_dir, exist_ok=True)

    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs")

    tasks = [
        (
            entity,
            project,
            run.id,
            run.name,
            output_dir,
            name_return,
            name_length,
        )
        for run in runs
    ]

    if num_workers is None:
        num_workers = min(8, mp.cpu_count())

    print(f"Using {num_workers} workers")

    with mp.get_context("spawn").Pool(num_workers) as pool:
        pool.map(process_single_run, tasks)


if __name__ == "__main__":
    download_all_wandb_histories_mp(
        entity="thomas-phd",
        project="SAC_ASYNC_TIP",
        output_dir="wandb_csv_exports",
        num_workers=18,
    )
