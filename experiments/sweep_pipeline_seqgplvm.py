from pathlib import Path
import subprocess, os, json, tempfile   
from itertools import product
from utils.pathing import as_path
from utils.training import dump_train_cfg_json
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
import numpy as np 
import pandas as pd
from utils.training import dump_train_cfg_json
from utils.checkpoints import make_train_id, build_training_parquet
from utils.training import class_to_id, tensor_fingerprint
import argparse


def run(cmd_list): subprocess.run(cmd_list, check=True)

dgp = "blackwell_yamauchi"

model_name = "seqgplvm"

dgp_index_path = as_path("./data/index/runs_monotone_included.parquet")

df_runs = pd.read_parquet(dgp_index_path)
for index, row in df_runs.iterrows():
    dic = json.loads(json.loads(row["config"]))
    for col in ["N", "T", "p", "a", "seed", "train_test_ratio","exclude_monotone"]:
        df_runs.at[index, col] = dic.get(col, None)

df_runs = df_runs[df_runs["dgp"]==dgp]
df_runs = df_runs[df_runs["exclude_monotone"]==False].reset_index(drop=True)

train_test_split = df_runs.loc[0,"train_test_ratio"]

#rho = [50] #[5,10,50]  # n/T

params_grid = {
    "n": [200], #n = [200, 500, 1000, 3000],
    "seed":[1],#sorted(list(df_runs.seed.unique())), # seed = [0,1,2,3,4]
    "a": [1],#,2], # a = [1,2]
    "p": [2],#,4], # p = [2,4]
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
    "optimize_hyperparams": {"lr": 1e-2, "num_epochs": 6000},
    "checkpoint_interval": 2000,
    "param_logging_freq": 50,
    "pid_col": "patient_id",
    "time_col": "t",
    "treatment_col": "D",
    "covariate_cols_prefix": "x",
    "x_standardize": True,
    "resume_mode": "no",
    "extra_logging": ["loss_list", "param_hist"], 
    "extra_logging_mode": "experiment",
    "drop_monotone": True, # whether to drop monotone rows during training, 
    "dgp_index_path": str(dgp_index_path), 
    "keep_checkpoints": False # whether to keep existing train/val checkpoints files in the output directory
}


num_inducing = {n: int(np.sqrt(train_test_split*n)) for n in params_grid["n"]}


device = "auto"

# Build the full grid of combos
full_grid = []
for n, seed, a, p, z_prior in product(*params_grid.values()):
    for t in df_runs[df_runs.N==n]["T"].drop_duplicates().sort_values():
        full_grid.append({"n": n, "seed": seed, "a": a, "p": p, "T": t, "z_prior": z_prior})
  
# ---------------------------------------------------------
# Parse optional chunk-size (how many configs per array task)
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--chunk-size", type=int, default=1)
args = parser.parse_args()
chunk_size = args.chunk_size

# Check if we are in a Slurm array
task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

if task_id is None:
    # Local / non-Slurm: run everything
    selected = full_grid
else:
    # Use 0-based array indices: --array=0-(NUM_CHUNKS-1)
    chunk_id = int(task_id)   # 0,1,2,...
    start = chunk_id * chunk_size
    end   = min(start + chunk_size, len(full_grid))

    if start >= len(full_grid):
        print(f"[sweep] chunk_id={chunk_id} start={start} >= grid size={len(full_grid)}; nothing to do.")
        raise SystemExit(0)

    selected = full_grid[start:end]
    print(f"[sweep] chunk_id={chunk_id}, processing configs [{start}:{end}) of {len(full_grid)}")


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

#print("GRID_SIZE =", len(full_grid))
#raise SystemExit

for combo in selected[:1]:
    n, seed, a, p, t, z_prior = int(combo["n"]), int(combo["seed"]), int(combo["a"]), int(combo["p"]), int(combo["T"]), combo["z_prior"]
    df_runs_subset = df_runs.query("dgp==@dgp and N==@n and T==@t and p==@p and seed==@seed and a==@a")
    
    dgp_cfg = json.loads(json.loads(df_runs_subset.iloc[0]["config"]))
    dgp_mani = json.loads(df_runs_subset.iloc[0]["manifest"])

    # Use task-specific temp files so array tasks don't overwrite each other
    stem = f"{dgp}_N{n}_T{t}_a{a}_p{p}_seed{seed}_zprior{z_prior}"
    dgp_cfg_path   = scratch / f"{stem}._data_tmp.json"
    dgp_mani_path  = scratch /f"{stem}._mani_tmp.json"
    train_cfg_path = scratch / f"{stem}._train_tmp.json"
    train_cfg_identity_path = scratch / f"{stem}._train_id_tmp.json"

    train_cfg_identity = {
        "N": dgp_cfg["N"],
        "T": dgp_cfg["T"],
        "C": dgp_cfg["p"] + training_cfg["treatment_lag"],  # total covariates including lagged treatments
        "latent_dim": training_cfg["latent_dim"],
        "num_inducing": num_inducing[n],
        "num_inducing_hidden": training_cfg["num_inducing_hidden"],
        "treatment_lag": training_cfg["treatment_lag"],
        "treatment_model": class_to_id(training_cfg["treatment_model"]),  
        "init_z": training_cfg["init_z"],  
        "z_prior": z_prior,
        "z_initializer": training_cfg["z_initializer"],
        "learn_inducing_locations": training_cfg["learn_inducing_locations"],
        "use_titsias": training_cfg["use_titsias"],
        "lr": training_cfg["optimize_hyperparams"]["lr"],
        "x_standardize": training_cfg["x_standardize"],
        "drop_monotone": training_cfg["drop_monotone"],
    }
    if training_cfg["z_initializer"] == "uniform":
        train_cfg_identity["uniform_halfwidth"] = training_cfg["uniform_halfwidth"]
    elif training_cfg["z_initializer"] == "normal":
        train_cfg_identity["prior_std"] = training_cfg["prior_std"]
    
    data_run_id = dgp_mani["run_id"]
    train_id = make_train_id(
        data_run_id=data_run_id,
        model_name=model_name,
        train_cfg=train_cfg_identity,
    )
    training_cfg["train_id"] = train_id

    try:
        dgp_cfg_path.write_text(json.dumps(dgp_cfg))
        dgp_mani_path.write_text(json.dumps(dgp_mani))
        training_cfg["uniform_halfwidth"] = a
        training_cfg["num_inducing"] = num_inducing[n]
        training_cfg["z_prior"] = z_prior
        dump_train_cfg_json(train_cfg_path, training_cfg)
        dump_train_cfg_json(train_cfg_identity_path, train_cfg_identity)

        run([
            "python", "-m", "experiments.pipeline_seqgplvm",
            "--dgp_config",   str(dgp_cfg_path),
            "--dgp_manifest", str(dgp_mani_path),
            "--train_cfg_identity", str(train_cfg_identity_path),
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
