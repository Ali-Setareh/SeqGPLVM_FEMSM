from itertools import product
import subprocess, json
from pathlib import Path

def run(cmd): subprocess.check_call(cmd, shell=True)

params_grid = {
    "n": [500, 1000],
    "T": [20, 40],
    "seed": [1, 2, 3], 
    "max_lag_d": 3,  # unused in this DGP
}
for n, T, seed in product(params_grid["n"], params_grid["T"], params_grid["seed"]):
    cfg = {"n": n, "T": T, "max_lag_d": 3, "seed": seed}
    cfg_path = Path("configs/_tmp.json")
    cfg_path.write_text(json.dumps(cfg))
    run(f"python experiments/run_simulation.py --dgp blackwell_yamauchi --config {cfg_path} --outdir data/raw")
