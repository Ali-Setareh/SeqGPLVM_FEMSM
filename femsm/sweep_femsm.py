import pandas as pd
from run_femsm import msm_estimates_from_df
import json 
from dgps import get_simulator
from pathlib import Path
import os
import argparse

df_runs = pd.read_parquet(Path(".")/"data"/"index"/"runs.parquet")
for index, row in df_runs.iterrows():
    dic = json.loads(json.loads(row["config"]))
    for col in ["N", "T", "p", "a", "seed", "train_test_ratio","exclude_monotone"]:
        df_runs.at[index, col] = dic.get(col, None)

df_runs = df_runs[df_runs["exclude_monotone"] == False].reset_index(drop=True)

parser = argparse.ArgumentParser()
parser.add_argument("--chunk-size", type=int, default=None)
parser.add_argument("--chunk-idx", type=int, default=None)
args = parser.parse_args()

# Figure out which chunk this process should handle
task_id = args.chunk_idx  # <- underscore, not minus

if task_id is None:
    tid_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tid_env is not None:
        task_id = int(tid_env)

if args.chunk_size is not None and task_id is not None:
    n = len(df_runs)
    chunk_size = args.chunk_size
    start = task_id * chunk_size
    end = min(start + chunk_size, n)
    df_runs = df_runs.iloc[start:end].reset_index(drop=True)
    print(f"Chunk {task_id}: handling rows [{start}:{end}) of {n}")
# else: df_runs stays unchanged → all runs processed

out_dir = Path("results") / "msm"
out_dir.mkdir(parents=True, exist_ok=True) 
# Append results to this file incrementally
backup_jsonl_suffix = f"_chunk{task_id}" if task_id is not None else ""
backup_jsonl = out_dir / f"fe_msm_results_incremental{backup_jsonl_suffix}.jsonl" 

#seed = 1 
#df_runs_seed = df_runs[df_runs["seed"]==seed].reset_index(drop=True)
all_rows = []
with backup_jsonl.open("a", encoding="utf-8") as backup_f:
    for index,rows in df_runs.iterrows():                     
        params = json.loads(json.loads(rows["config"]))
        mani = json.loads(rows["manifest"])
        simulate = get_simulator(params["dgp"])
        df_sim = simulate(params)

        with open(Path(".") / mani["split_file"], "r", encoding="utf-8") as f:
            split_data = json.load(f)

        train_ids = split_data["train_ids"]              
        x_cols = [c for c in df_sim.columns if c.startswith("x")] 

        row = msm_estimates_from_df(
            df_sim,
            train_ids,
            k_last=4,
            a_val=params.get("a", None),
            data_id=mani.get("run_id", None),
            x_cols=x_cols,                        # optional
        )
        
        all_rows.append(row)
        row["seed"] = params.get("seed", None)
        print(F"{index}/{len(df_runs)}: N: {params.get('N', None)} , T: {params.get('T', None)} , a: {params.get('a', None)} , p: {params.get('p', None)}, seed: {params.get('seed', None)} done. \n {row}", end="\r")
        row.to_json(backup_f, orient="records", lines=True)
        backup_f.flush()
    

results = pd.concat(all_rows, ignore_index=True)
# Use chunk id for filename so parallel jobs don't overwrite each other
chunk_suffix = f"_chunk{task_id}" if task_id is not None else ""
results.to_csv(out_dir / f"fe_msm_results{chunk_suffix}.csv", index=False)

