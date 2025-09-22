from __future__ import annotations
import json, hashlib, platform, subprocess
from pathlib import Path
from datetime import datetime

def canonicalize(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def make_run_id(params: dict, length: int = 8) -> str:
    s = canonicalize(params)
    h = hashlib.blake2s(s.encode("utf-8"), digest_size=16).hexdigest()
    return h[:length]

def run_dir(root: Path, dgp: str, run_id: str) -> Path:
    return root / "data" / "raw" / dgp / run_id

def save_dataset_run(root: Path, dgp: str, params: dict, df, *, extra_manifest: dict | None = None):
    root = Path(root)
    rid = make_run_id(params)
    out = run_dir(root, dgp, rid)
    out.mkdir(parents=True, exist_ok=True)

    # 1) config used for generation
    (out / "config.json").write_text(canonicalize(params), encoding="utf-8")

    # 2) data
    df.to_parquet(out / "data.parquet", index=False)

    # 3) manifest.json (lightweight, human friendly)
    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root).decode().strip()
    except Exception:
        pass

    manifest = {
        "dgp": dgp,
        "run_id": rid,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "path": str(out),
        "git_commit": git_commit,
        "python": platform.python_version(),
        "node": platform.node(),
        "params": params,  # full dict (small; if huge, store only in config.json)
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rid, out, manifest

def append_global_index(root: Path, manifest_row: dict):
    import pandas as pd
    idx_path = Path(root) / "data" / "index" / "runs.parquet"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    row = manifest_row.copy()

    # explode a few common params for filtering
    params = row.get("params", {})
    for k in ("N", "n", "T", "K", "p","a" ,"seed", "split_seed"):
        if k in params:
            row[k] = params[k]

    if idx_path.exists():
        df = pd.read_parquet(idx_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_parquet(idx_path)

def find_by_params(root: Path, dgp: str, query_params: dict):
    """Exact match via hash; fallback to column filter."""
    import pandas as pd
    idx_path = Path(root) / "data" / "index" / "runs.parquet"
    if not idx_path.exists():
        return None
    df = pd.read_parquet(idx_path)
    df = df[df["dgp"] == dgp]
    target = make_run_id(query_params)
    hit = df[df["run_id"] == target]
    if not hit.empty:
        return hit.iloc[0].to_dict()
    # fallback: filter by keys that we exploded
    for k in ("N","n","T","K","p","a","seed","split_seed"):
        if k in query_params and k in df.columns:
            df = df[df[k] == query_params[k]]
    return df.iloc[0].to_dict() if not df.empty else None


def load_by_run_id(root: str | Path, dgp: str, run_id: str, *, columns=None):
    """Load data.parquet + manifest for a known run_id."""
    import pandas as pd, json
    root = Path(root)
    run_path = run_dir(root, dgp, run_id)
    df = pd.read_parquet(run_path / "data.parquet", columns=columns)
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    return df, manifest

def load_by_params(root: str | Path, params: dict, *, columns=None):
    """Find the run via params (hash) and load its dataframe."""
    hit = find_by_params(root, params["dgp"], params)
    if not hit:
        raise FileNotFoundError(f"No indexed run for dgp={params["dgp"]} with params={params}")
    return load_by_run_id(root, params["dgp"], hit["run_id"], columns=columns)
