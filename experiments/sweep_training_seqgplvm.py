from pathlib import Path
import subprocess, os, json, tempfile   
from itertools import product
from utils.training import dump_train_cfg_json
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
import numpy as np 

def run(cmd_list): subprocess.run(cmd_list, check=True)

dgp = "blackwell_yamauchi"

rho = [50] #[5,10,50]  # n/T
params_grid = {
    "n": [200], #n = [200, 500, 1000, 3000],
    "seed": [1],
    "a": [1], # a = [1,2]
    "p": [2], # p = [2,4]
    "z_prior": ["normal"] # [normal, uniform] hidden confounder prior types, only normal for now becaue the KL term for uniform prior is not implemented
}


train_test_split = 0.8
T = {((1/train_test_split) * n): [int(n/r) for r in rho] for n in params_grid["n"]}
num_inducing = {((1/train_test_split) * n): int(np.sqrt(n)) for n in params_grid["n"]}
params_grid["n"] = [(1/train_test_split) * i for i in params_grid["n"]]

beta_dict  = {2: [-0.5, -0.5],           4: [-0.5, -0.5, 1.0, -0.5]}
gamma_dict = {2: [1.0, 0.5],             4: [1.0, 0.5, 1.0, 1.0]}

params = dict(
    phi=0.3, tau_F=1.0, tau_C=0.3, mean_x=-0.5, offdiag=0.2,
    sigma_eps=1.0, max_lag_x=0, max_lag_d=3, split_seed=42,
    treatment_model="logit",
    exclude_monotone=True
)

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
    "resume_mode": "yes",
}

device = "auto"

# Build the full grid of combos
full_grid = []
for n, seed, a, p, z_prior in product(*params_grid.values()):
    for t in T[n]:
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

# Ensure project-local configs dir exists (for logs/checks, optional)
#Path("configs").mkdir(exist_ok=True)

for combo in selected:
    n, seed, a, p, t, z_prior = combo["n"], combo["seed"], combo["a"], combo["p"], combo["T"], combo["z_prior"]

    dgp_cfg = {"dgp": dgp,"N": n, "T": t, "train_test_ratio": train_test_split,"seed": seed, "a": a, "p": p,
               "beta": beta_dict[p], "gamma": gamma_dict[p], **params}

    # Use task-specific temp files so array tasks don't overwrite each other
    stem = f"{dgp}_N{n}_T{t}_a{a}_p{p}_seed{seed}_zprior{z_prior}"
    dgp_cfg_path   = scratch / f"{stem}._data_tmp.json"
    train_cfg_path = scratch / f"{stem}._train_tmp.json"

    try:
        dgp_cfg_path.write_text(json.dumps(dgp_cfg))
        training_cfg["uniform_halfwidth"] = a
        training_cfg["num_inducing"] = num_inducing[n]
        training_cfg["z_prior"] = z_prior
        dump_train_cfg_json(train_cfg_path, training_cfg)

        run([
            "python", "-m", "experiments.train_seqgplvm",
            "--data",   str(dgp_cfg_path),
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
