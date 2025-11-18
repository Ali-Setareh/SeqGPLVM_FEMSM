from pathlib import Path
import subprocess, os, json, tempfile   
from itertools import product
from utils.training import dump_train_cfg_json
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
import numpy as np 
import pandas as pd

def run(cmd_list): subprocess.run(cmd_list, check=True)

dgp = "blackwell_yamauchi"

df_runs = pd.read_parquet(Path(".")/"data"/"index"/"runs.parquet")
for index, row in df_runs.iterrows():
    dic = json.loads(json.loads(row["config"]))
    for col in ["N", "T", "p", "a", "seed", "train_test_ratio","exclude_monotone"]:
        df_runs.at[index, col] = dic.get(col, None)

df_runs = df_runs[df_runs["dgp"]==dgp]
train_test_split = df_runs.loc[0,"train_test_ratio"]

#rho = [50] #[5,10,50]  # n/T

params_grid = {
    "n": [200], #n = [200, 500, 1000, 3000],
    "seed": [1],#list(df_runs.seed.unique()), 
    "a": [1], # a = [1,2]
    "p": [2], # p = [2,4]
    "z_prior": ["normal"] # [normal, uniform] hidden confounder prior types, only normal for now becaue the KL term for uniform prior is not implemented
}

params_grid["n"] = [(1/train_test_split * n) for n in params_grid["n"]]


training_cfg = {
    "latent_dim": 1,
    "num_inducing_hidden": 13,
    "treatment_lag": 1,
    "init_z": None,
    "z_initializer": "uniform",
    "uniform_halfwidth": 2.0,
    "treatment_model": BernoulliLikelihood,
    "learn_inducing_locations": False,
    "use_titsias": False,
    "optimize_hyperparams": {"lr": 1e-2, "num_epochs": 200},
    "checkpoint_interval": 200,
    "param_logging_freq": 50,
    "pid_col": "patient_id",
    "time_col": "t",
    "treatment_col": "D",
    "covariate_cols_prefix": "x",
    "x_standardize": True,
    "resume_mode": "no",
    "extra_logging": ["loss_list"], 
    "extra_logging_mode": "experiment"
}


num_inducing = {n: int(np.sqrt(train_test_split*n)) for n in params_grid["n"]}


device = "auto"

# Build the full grid of combos
full_grid = []
for n, seed, a, p, z_prior in product(*params_grid.values()):
    for t in df_runs[df_runs.N==n]["T"].drop_duplicates().sort_values():
        full_grid.append({"n": n, "seed": seed, "a": a, "p": p, "T": t, "z_prior": z_prior})
  
# Check if we are in a Slurm array
task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
if task_id is None:
    # Fallback: run all combos sequentially (as before)
    selected = full_grid
else:
    idx = int(task_id) - 1  # Slurm arrays are 1-based
    if not (0 <= idx < len(full_grid)):
        raise SystemExit(f"Task id {task_id} out of range (grid size={len(full_grid)})")
    selected = [full_grid[idx]]

# Scratch selection: Slurm uses TMPDIR; local uses repo configs/ and deletes after run
is_slurm = task_id is not None
if is_slurm:
    job_id   = os.environ.get("SLURM_JOB_ID", "nojid")
    task_tag = os.environ.get("SLURM_ARRAY_TASK_ID", "single")
    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()   # ← cross-platform fallback
    scratch  = Path(tmp_root) / f"sweep_{job_id}_{task_tag}"
    scratch.mkdir(parents=True, exist_ok=True)
    delete_after = False
else:
    scratch = Path("configs")
    scratch.mkdir(exist_ok=True)
    delete_after = True

for combo in selected:
    n, seed, a, p, t, z_prior = int(combo["n"]), int(combo["seed"]), int(combo["a"]), int(combo["p"]), int(combo["T"]), combo["z_prior"]
    df_runs_subset = df_runs.query("dgp==@dgp and N==@n and T==@t and p==@p and seed==@seed and a==@a")
    
    dgp_cfg = json.loads(json.loads(df_runs_subset.iloc[0]["config"]))
    dgp_mani = json.loads(df_runs_subset.iloc[0]["manifest"])

    # Use task-specific temp files so array tasks don't overwrite each other
    stem = f"{dgp}_N{n}_T{t}_a{a}_p{p}_seed{seed}_zprior{z_prior}"
    dgp_cfg_path   = scratch / f"{stem}._data_tmp.json"
    dgp_mani_path  = scratch / f"{stem}._mani_tmp.json"
    train_cfg_path = scratch / f"{stem}._train_tmp.json"


    try:
        dgp_cfg_path.write_text(json.dumps(dgp_cfg))
        dgp_mani_path.write_text(json.dumps(dgp_mani))
        training_cfg["uniform_halfwidth"] = a
        training_cfg["num_inducing"] = num_inducing[n]
        training_cfg["z_prior"] = z_prior
        dump_train_cfg_json(train_cfg_path, training_cfg)

        run([
            "python", "-m", "experiments.train_seqgplvm",
            "--dgp_config",   str(dgp_cfg_path),
            "--dgp_manifest", str(dgp_mani_path),
            "--config", str(train_cfg_path),
            "--device", device
        ])
    finally:
        if delete_after:
                # keep local workspace clean
                try: dgp_cfg_path.unlink(missing_ok=True)
                except Exception: pass
                try: train_cfg_path.unlink(missing_ok=True)
                except Exception: pass
