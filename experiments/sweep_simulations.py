from itertools import product
import subprocess, json
from pathlib import Path

def run(cmd): subprocess.check_call(cmd, shell=True)

dgp = "blackwell_yamauchi"

rho = [5,10,50] # n/T
params_grid = {
    "n": [200, 500, 1000, 3000],
    "seed": [1], 
    "a": [1,2], 
    "p": [2,4], 
}

T =  {n:[int(n/r) for r in rho] for n in params_grid["n"]}

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

for n, seed, a, p in product(*params_grid.values()):
    for t in T[n]:
        cfg = {"dgp": dgp,"n": n, "T": t, "seed": seed, "a": a, "p": p,
               "beta": beta_dict[p], "gamma": gamma_dict[p], **params}
    
        cfg_path = Path("configs/_tmp.json")
        cfg_path.write_text(json.dumps(cfg))
        cmd = [
        "python", "experiments/run_simulation.py",
        "--dgp", dgp,
        "--config", str(cfg_path),
        "--project_root", ".",
        "--splits_outdir", f"data/splits/{dgp}",
        ]
        run(cmd)
