import pandas as pd
from run_femsm import msm_estimates_from_df
import json 
from dgps import get_simulator
from pathlib import Path


df_runs = pd.read_parquet(Path(".")/"data"/"index"/"runs.parquet")
for index, row in df_runs.iterrows():
    dic = json.loads(json.loads(row["config"]))
    for col in ["N", "T", "p", "a", "seed", "train_test_ratio","exclude_monotone"]:
        df_runs.at[index, col] = dic.get(col, None)

#seed = 1 
#df_runs_seed = df_runs[df_runs["seed"]==seed].reset_index(drop=True)
all_rows = []
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
    

results = pd.concat(all_rows, ignore_index=True)
out_dir = Path("results") / "msm"
out_dir.mkdir(parents=True, exist_ok=True) 
results.to_csv(out_dir / "fe_msm_results.csv", index=False)
