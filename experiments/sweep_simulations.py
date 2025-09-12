from itertools import product
import subprocess, json
from pathlib import Path

def run(cmd): subprocess.check_call(cmd, shell=True)

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

for n, T, seed, a, p in product(*params_grid.values()):
    cfg = {"n": n, "T": T, "seed": seed, "a": a, "p": p,
           "beta": beta_dict[p], "gamma": gamma_dict[p], **params}
    
    cfg_path = Path("configs/_tmp.json")
    cfg_path.write_text(json.dumps(cfg))
    run(f"python experiments/run_simulation.py --dgp blackwell_yamauchi --config {cfg_path} --outdir data/raw/{dgp}")
