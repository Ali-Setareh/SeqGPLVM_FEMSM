# sweep_training.py
from pathlib import Path
import subprocess, os, json
from itertools import product


def run(cmd_list): subprocess.run(cmd_list, check=True)

dgp = "blackwell_yamauchi"

rho = [5,10,50]  # n/T
params_grid = {
    "n": [200], #n = [200, 500, 1000, 3000],
    "seed": [1],
    "a": [1,2],
    "p": [2,4],
}

T = {n: [int(n/r) for r in rho] for n in params_grid["n"]}
beta_dict  = {2: [-0.5, -0.5],           4: [-0.5, -0.5, 1.0, -0.5]}
gamma_dict = {2: [1.0, 0.5],             4: [1.0, 0.5, 1.0, 1.0]}

params = dict(
    phi=0.3, tau_F=1.0, tau_C=0.3, mean_x=-0.5, offdiag=0.2,
    sigma_eps=1.0, max_lag_x=0, max_lag_d=3, split_seed=42,
    treatment_model="logit",
)

training_cfg = {
    "latent_dim": 1,
    "num_inducing": 50,
    "num_inducing_hidden": 5,
    "treatment_lag": 1,
    "optimize_hyperparams": {"lr": 1e-2, "num_epochs": 20000},
    "checkpoint_interval": 2000,
    "param_logging_freq": 50,
    "pid_col": "patient_id",
    "time_col": "t",
    "treatment_col": "D",
    "covariate_cols_prefix": "x",
    "resume_mode": "no",
}

device = "auto"

# Build the full grid of combos
full_grid = []
for n, seed, a, p in product(*params_grid.values()):
    for t in T[n]:
        full_grid.append({"n": n, "seed": seed, "a": a, "p": p, "T": t})

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

# Unique scratch dir per task to avoid config filename clashes
job_id   = os.environ.get("SLURM_JOB_ID", "nojid")
task_tag = os.environ.get("SLURM_ARRAY_TASK_ID", "single")
tmp = os.environ.get("TMPDIR", "/tmp")
scratch = Path(tmp) / f"sweep_{job_id}_{task_tag}"
scratch.mkdir(parents=True, exist_ok=True)

# Ensure project-local configs dir exists (for logs/checks, optional)
Path("configs").mkdir(exist_ok=True)

for combo in selected:
    n, seed, a, p, t = combo["n"], combo["seed"], combo["a"], combo["p"], combo["T"]

    dgp_cfg = {
        "dgp": dgp, "n": n, "T": t, "seed": seed, "a": a, "p": p,
        "beta": beta_dict[p], "gamma": gamma_dict[p], **params
    }

    # Use task-specific temp files so array tasks don't overwrite each other
    stem = f"{dgp}_N{n}_T{t}_a{a}_p{p}_seed{seed}"
    dgp_cfg_path   = scratch / f"{stem}._data_tmp.json"
    train_cfg_path = scratch / f"{stem}._train_tmp.json"

    dgp_cfg_path.write_text(json.dumps(dgp_cfg))
    train_cfg_path.write_text(json.dumps(training_cfg))

    run([
        "python", "-m", "experiments.train_seqgplvm",
        "--data",   str(dgp_cfg_path),
        "--config", str(train_cfg_path),
        "--device", device
    ])
