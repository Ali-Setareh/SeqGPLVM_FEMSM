import argparse, json
from pathlib import Path
import numpy as np
from dgps import get_simulator
from dgps.base import make_stem 
from utils.splits import make_or_load_split, split_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dgp", required=True, help="module name in dgps/, e.g. hatt_feuerriegel")
    p.add_argument("--config", required=True, help="JSON or YAML with params")
    p.add_argument("--outdir", default="data/raw")
    p.add_argument("--splits_outdir", default="data/splits")

    args = p.parse_args()

    # Read params
    cfg_path = Path(args.config)
    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        import yaml
        params = yaml.safe_load(cfg_path.read_text())
    else:
        params = json.loads(cfg_path.read_text())

    # Seed handling
    seed = params.get("seed", 0)
    params["seed"] = seed

    # Simulate
    simulate = get_simulator(args.dgp)
    df = simulate(params)

    # Save data + metadata
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits_outdir = Path(args.splits_outdir)
    splits_outdir.mkdir(parents=True, exist_ok=True)
    stem = make_stem(args.dgp, params)
    df.to_parquet(outdir / f"{stem}.parquet", index=False)

    meta = {
        "dgp": args.dgp,
        "params": params,
        "script": "experiments/run_simulation.py",
        "data_file":f"{str(outdir)} /{stem}.parquet",
    }

    N = int(params["n"])
    T = int(params.get("T", 0)) or None
    p = int(params.get("p", 0)) or None
    split_seed = int(params.get("split_seed", 42))   # optional, separate from data seed

    _,split_file = make_or_load_split(dgp = args.dgp, N = N, split_seed = split_seed, T=T, p=p, output_dir=splits_outdir)
    

    # add to metadata you already write:
    meta["split_file"] = str(split_file)
    meta["split_info"] = {"by": "unit", "split_seed": split_seed}

    (outdir / f"{stem}.metadata.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
