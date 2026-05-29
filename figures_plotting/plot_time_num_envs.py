import pandas as pd

df = pd.read_csv("wandb_csv_exports/num_envs_seed_time_25000_steps.csv")

stats = df.groupby("num_envs")["time"].agg(["mean", "std"]).sort_index()

print("Mean ± std runtime (seconds):\n")
for num_envs, row in stats.iterrows():
    print(f"{num_envs:>2} envs : {row['mean']:.2f} ± {row['std']:.2f}")
