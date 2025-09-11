import argparse, json
from pathlib import Path
import numpy as np
from dgps import get_simulator

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dgp", required=True, help="module name in dgps/, e.g. hatt_feuerriegel")
    p.add_argument("--config", required=True, help="JSON or YAML with params")
    p.add_argument("--outdir", default="data/raw")
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
    rng = np.random.default_rng(seed)
    params["seed"] = seed

    # Simulate
    simulate = get_simulator(args.dgp)
    df = simulate(params, rng=rng)

    # Save data + metadata
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.dgp}_n{params["n"]}_T{params["T"]}_seed{seed}'
    df.to_parquet(outdir / f"{stem}.parquet", index=False)

    meta = {
        "dgp": args.dgp,
        "params": params,
        "script": "experiments/run_simulation.py",
        "data_file": f"{stem}.parquet",
    }
    (outdir / f"{stem}.metadata.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
