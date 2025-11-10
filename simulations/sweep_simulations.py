from itertools import product
import subprocess, json
from pathlib import Path
import os, argparse, hashlib


def write_unique_cfg(cfg: dict, index: int) -> Path:
    # Prefer node-local scratch on HPC; fall back to repo dir locally
    tmpdir = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "configs/_tmp"
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Make a short, deterministic name from the cfg content + task identity
    h = hashlib.blake2s(json.dumps(cfg, sort_keys=True).encode(), digest_size=8).hexdigest()
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "notask")
    pid = os.getpid()
    cfg_path = tmpdir / f"cfg_{index}_{task}_{pid}_{h}.json"

    cfg_path.write_text(json.dumps(cfg))
    return cfg_path

def _flatten(row: dict) -> dict:
    """Turn nested dicts into JSON strings; ensure stable keys."""
    import json as _json
    out = dict(row)  # shallow copy
    # stringify nested dicts (schema-stable, fast)
    for k in ("manifest", "config"):
        v = out.get(k)
        out[k] = _json.dumps(v) if v is not None else None
    # make sure all expected keys exist
    for k in ("dgp","run_id","treatment_model","path","created_at",
              "git_commit","split_file","manifest","config","replay_command"):
        out.setdefault(k, None)
    return out

def _write_parquet(rows, out: str):
    import pyarrow as pa, pyarrow.parquet as pq
    schema = pa.schema([
        ("dgp", pa.string()),
        ("run_id", pa.string()),
        ("treatment_model", pa.string()),
        ("path", pa.string()),
        ("created_at", pa.string()),
        ("git_commit", pa.string()),
        ("split_file", pa.string()),
        ("manifest", pa.large_string()),
        ("config", pa.large_string()),
        ("replay_command", pa.large_string()),
    ])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)  # single big batch = fastest here
    pq.write_table(table, out, compression="zstd")     # compact & fast


def run(cmd):
    # capture stdout so we can parse the JSON line
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


dgp = "blackwell_yamauchi"

rho = [5,10,50] # n/T
params_grid = {
    "n": [200, 500, 1000, 3000], 
    "seed": list(range(1,101)), 
    "a": [1,2], 
    "p": [2,4], 
}

train_test_split = 0.8 

T =  {int((1/train_test_split) * n):[int(n/r) for r in rho] for n in params_grid["n"]}
params_grid["n"] = [int(i * (1/train_test_split)) for i in params_grid["n"]]  # we have to adjust n so that after an 80/20 split we get the desired n

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
params["exclude_monotone"] = True

index = 0
total =  len(params_grid["n"]) * len(params_grid["seed"]) * len(params_grid["a"]) * len(params_grid["p"]) * len(rho)
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=None,
                help="Run only this 0-based global index (for SLURM_ARRAY_TASK_ID).")
ap.add_argument("--defer-index", action="store_true",
                help="Pass --index_mode=deferred to run_simulation and rebuild later.")
ap.add_argument("--count", action="store_true",
                help="Print total number of tasks and exit.")
args = ap.parse_args()

if args.count:
    print(total)
    raise SystemExit

# if launched as array without --task, pick it up automatically
if args.task is None:
    env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_task is not None:
        args.task = int(env_task)

rows = [] 

for n, seed, a, p in product(*params_grid.values()):

    for t in T[n]:
        cfg = {"dgp": dgp,"N": n, "T": t, "train_test_ratio": train_test_split,"seed": seed, "a": a, "p": p,
               "beta": beta_dict[p], "gamma": gamma_dict[p], **params}
        
        # Select by global index if requested
        if args.task is not None and index != args.task:
            index += 1
            continue
    
        cfg_path = write_unique_cfg(cfg, index)
        cmd = [
        "python", "simulations/run_simulation.py",
        "--dgp", dgp,
        "--config", str(cfg_path.resolve()),
        "--project_root", ".",
        "--splits_outdir", f"data/splits/{dgp}/",
        "--index_mode", "deferred", 
        "--emit-row-stdout",  
        ]
        
        if args.defer_index:
            cmd += ["--index_mode", "deferred"]

        line = run(cmd)              # returns the child's stdout (JSON line)
        row = json.loads(line)       # dict
        rows.append(_flatten(row))   # stringify nested fields for Parquet

        print(f"[{index+1}/{total}]", " ".join(cmd), flush=True)
        index += 1
        run(cmd)
        try:
            if not os.environ.get("SLURM_TMPDIR"):
                cfg_path.unlink(missing_ok=True)
        except Exception:
            pass
_write_parquet(rows, out="data/index/runs.parquet")

