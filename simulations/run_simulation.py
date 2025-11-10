# experiments/run_simulation.py
import functools
import argparse, json
from pathlib import Path
import sys

import numpy as np
from dgps import get_simulator
from utils.splits import make_or_load_split
from utils.runs import save_dataset_run, append_global_index, make_run_id, canonicalize

def main():

    print = functools.partial(print, file=sys.stderr)

    p = argparse.ArgumentParser()
    p.add_argument("--dgp", required=True, help="module name in dgps/, e.g. hatt_feuerriegel")
    p.add_argument("--config", required=True, help="JSON or YAML with params")
    p.add_argument("--project_root", default=".", help="repo root (where data/ lives)")
    p.add_argument("--splits_outdir", default="data/splits", help="dir for persisted splits")
    p.add_argument("--save_data", choices=["full","head","none"], default="none",
                   help="full: save complete dataset; head: save preview; none: save nothing")
    p.add_argument("--head_k", type=int, default=500, help="rows to keep if save_data=head")
    p.add_argument("--index_mode", choices=["append", "deferred"], default="append",
               help="append: update runs.parquet immediately; deferred: skip (rebuild later)")
    p.add_argument("--write_config_manifest", action="store_true", help="Whether to write config.json and manifest.json files")
    p.add_argument("--rowlog", type=str, default=None,
               help="Path to a JSONL file to append one summary row per run.")

    
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
    if args.save_data == "none":
        df = None 
    else:
        simulate = get_simulator(args.dgp)
        df = simulate(params)

    # --- Save canonical dataset run (short ID folder) ---
    extra_manifest = {
        "script": "simulations/run_simulation.py",
        "save_mode": args.save_data,
        "rng_info": {"lib": "numpy", "version": np.__version__, "seed": params["seed"], "rng_source": "rng_from_seed"},
    }

    run_id, run_path, manifest, config = save_dataset_run(root, args.dgp, params, df, 
                                                  extra_manifest=extra_manifest, 
                                                  save_mode=args.save_data,
                                                  head_k=args.head_k, write_config_manifest=args.write_config_manifest)

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
        "replay_command": (
            f"python simulations/run_simulation.py --dgp {args.dgp} "
            f"--config {args.config} --project_root {args.project_root} "
            f"--splits_outdir {args.splits_outdir} --save_data {args.save_data}"
        ),
    })

    if args.write_config_manifest:

        (run_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- Global index row ---
    row = {
        "dgp": args.dgp,
        "run_id": run_id,
        "treatment_model": params.get("treatment_model"),
        "path": str(run_path),
        "created_at": manifest["created_at"],
        "git_commit": manifest.get("git_commit"),
        "split_file": str(split_file),
        "manifest": manifest,
        "config": config,
        "replay_command": manifest["replay_command"],
    }
    if args.index_mode == "append" and not args.rowlog:
        append_global_index(root, row)
    

    if args.rowlog:
        def _json_default(o):
            import numpy as _np
            from pathlib import Path as _Path
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, _Path):
                return str(o)
            return str(o)
        rowlog_path = Path(args.rowlog)
        rowlog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rowlog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")



    if args.save_data != "none":
        print(f"[OK] dgp={args.dgp} run_id={run_id}")
        print(f"     data: {run_path/'data.parquet'}")
    else:
        print(f"[OK] dgp={args.dgp} run_id={run_id} (no dataset saved)")
    
    if args.write_config_manifest:
        print(f"     cfg : {run_path/'config.json'}")
        print(f"     mani: {run_path/'manifest.json'}")
        print(f"     split: {split_file}")

if __name__ == "__main__":
    main()