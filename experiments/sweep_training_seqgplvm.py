from pathlib import Path
import subprocess
from itertools import product
from dgps.base import make_stem
import json


def run(cmd_list): subprocess.run(cmd_list, check=True)

dgp = "blackwell_yamauchi"

params_grid = {
    "n": [500],
    "T": [20],
    "seed": [1], 
    "a": [1.0], 
    "p": [2], 
}

beta_dict = {2:  [-0.5, -0.5], 4: [-0.5, -0.5, 1.0, -0.5]}
gamma_dict = {2: [1.0, 0.5], 4: [1.0, 0.5, 1.0, 1.0]}

params = {
    "phi": 0.3,
    "tau_F": 1.0, 
    "tau_C": 0.3,
    "mean_x": -0.5, 
    "offdiag": 0.2,
    "sigma_eps": 1.0, 
    "max_lag_x": 0, 
    "max_lag_d": 3,
    "split_seed": 42
    }

treatment_model = "logit"

params["treatment_model"] = treatment_model


training_cfg = {"latent_dim": 1,
                "num_inducing": 50,
                "num_inducing_hidden": 5,
                "treatment_lag": 1,
                "optimize_hyperparams": {"lr": 1e-2, "num_epochs": 100},
                "checkpoint_interval": 50,
                "param_logging_freq": 10,
                "pid_col": "patient_id",
                "time_col": "t",
                "treatment_col": "D",
                "covariate_cols_prefix": "x"
}

training_cfg["resume_mode"] = "no"

device = "auto"



for n, T, seed, a, p in product(*params_grid.values()):
    dgp_cfg = {"dgp": dgp, "n": n, "T": T, "seed": seed, "a": a, "p": p,
           "beta": beta_dict[p], "gamma": gamma_dict[p], **params}
    dgp_cfg_path = Path("configs/_data_tmp.json")
    dgp_cfg_path.write_text(json.dumps(dgp_cfg))

    train_cfg_path = Path("configs/_train_tmp.json")
    train_cfg_path.write_text(json.dumps(training_cfg))
   
    run([
        "python", "-m", "experiments.train_seqgplvm",
        "--data", dgp_cfg_path,
        "--config", train_cfg_path,
        "--device", device
    ])
