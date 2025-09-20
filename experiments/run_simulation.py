# experiments/run_simulation.py
import argparse, json
from pathlib import Path

import numpy as np
from dgps import get_simulator
from utils.splits import make_or_load_split
from utils.runs import save_dataset_run, append_global_index, make_run_id, canonicalize

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dgp", required=True, help="module name in dgps/, e.g. hatt_feuerriegel")
    p.add_argument("--config", required=True, help="JSON or YAML with params")
    p.add_argument("--project_root", default=".", help="repo root (where data/ lives)")
    p.add_argument("--splits_outdir", default="data/splits", help="dir for persisted splits")
    
    args = p.parse_args()

    root = Path(args.project_root)

    # --- Load params (JSON or YAML) ---
    cfg_path = Path(args.config)
    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        import yaml
        params = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        params = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Normalize/ensure seed present
    params["seed"] = int(params.get("seed", 0))

    # --- Simulate ---
    simulate = get_simulator(args.dgp)
    df = simulate(params)

    # --- Save canonical dataset run (short ID folder) ---
    extra_manifest = {"script": "experiments/run_simulation.py"}
    run_id, run_path, manifest = save_dataset_run(root, args.dgp, params, df, extra_manifest=extra_manifest)

    # --- Splits (persist + record path in manifest) ---
    splits_outdir = root / args.splits_outdir
    splits_outdir.mkdir(parents=True, exist_ok=True)

    N = int(params.get("N", params.get("n")))
    T = int(params.get("T", 0)) or None
    p_dim = int(params.get("p", 0)) or None
    split_seed = int(params.get("split_seed", 42))

    
    _, split_file = make_or_load_split(
        dgp=args.dgp, N=N, split_seed=split_seed, T=T, p=p_dim, output_dir=splits_outdir
    )

    # Update manifest.json with split info
    manifest.update({
        "split_file": str(split_file),
        "split_info": {"by": "unit", "split_seed": split_seed},
    })
    (run_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- Global index row ---
    row = {
        "dgp": args.dgp,
        "run_id": run_id,
        "path": str(run_path),
        "created_at": manifest["created_at"],
        "git_commit": manifest.get("git_commit"),
        "params": params,
        "split_file": str(split_file),
    }
    append_global_index(root, row)


    print(f"[OK] dgp={args.dgp} run_id={run_id}")
    print(f"     data: {run_path/'data.parquet'}")
    print(f"     cfg : {run_path/'config.json'}")
    print(f"     mani: {run_path/'manifest.json'}")
    print(f"     split: {split_file}")

if __name__ == "__main__":
    main()
