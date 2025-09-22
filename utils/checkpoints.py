import os 
import glob
from datetime import datetime
from pathlib import Path 
import json, platform, subprocess
import hashlib
import torch 

def canonicalize(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def short_hash(obj, length: int = 8) -> str:
    s = canonicalize(obj)
    return hashlib.blake2s(s.encode("utf-8"), digest_size=16).hexdigest()[:length]

def make_train_id(*, data_run_id: str, model_name: str, train_cfg: dict, extra: dict | None = None, length: int = 10) -> str:
    payload = {"data": data_run_id, "model": model_name, "cfg": train_cfg}
    if extra: payload["extra"] = extra
    return short_hash(payload, length=length)

def train_dir(root: Path | str, model_name: str, train_id: str) -> Path:
    return Path(root) / "results" / "models" / model_name / train_id

def write_train_files(root: Path | str,
                      model_name: str,
                      train_id: str,
                      *,
                      train_cfg: dict,
                      data_ref: dict,
                      metrics: dict | None = None):
    root = Path(root)
    out = train_dir(root, model_name, train_id)
    (out / "ckpts").mkdir(parents=True, exist_ok=True)

    # Save configs / refs
    (out / "config.json").write_text(canonicalize(train_cfg), encoding="utf-8")
    (out / "data_ref.json").write_text(canonicalize(data_ref), encoding="utf-8")

    # manifest
    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root).decode().strip()
    except Exception:
        pass
    manifest = {
        "model": model_name,
        "train_id": train_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": git_commit,
        "python": platform.python_version(),
        "node": platform.node(),
        "data_ref": data_ref,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return out

def save_ckpt(path: Path, step: int, model_state: dict, optimizer_state: dict | None = None, extra: dict | None = None):
    payload = {"model_state": model_state}
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra:
        payload["extra"] = extra
    ckpt_path = path / "ckpts" / f"step_{step:07d}.pt"
    import torch
    torch.save(payload, ckpt_path)
    return ckpt_path

######### Indexing Trainings #########
def _explode_for_filtering(row: dict, keys: tuple[str, ...], source: dict, prefix: str | None = None):
    for k in keys:
        if k in source:
            row[(f"{prefix}_" if prefix else "") + k] = source[k]

def read_manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

def append_training_index(root: str | Path, row: dict):
    """
    Append a row to data/index/training.parquet. Creates file if absent.
    Row MUST include: model, train_id, data_run_id, path, created_at.
    """
    import pandas as pd
    idx_path = Path(root) / "data" / "index" / "training.parquet"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    if idx_path.exists():
        df = pd.read_parquet(idx_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_parquet(idx_path)

def make_training_index_row(
    root: str | Path,
    model_name: str,
    train_id: str,
    train_cfg: dict,
    *,
    data_run_id: str,
    metrics: dict | None = None,
) -> dict:
    """
    Build a tidy index row from files on disk + inputs.
    """
    out = train_dir(root, model_name, train_id)
    mani = read_manifest(out)  # created by write_train_files(...)
    row = {
        "model": model_name,
        "train_id": train_id,
        "data_run_id": data_run_id,
        "path": str(out),
        "created_at": mani["created_at"],
        "git_commit": mani.get("git_commit"),
    }
    # explode a few train_cfg keys for easy filtering
    keys_to_explode = (
        "latent_dim", "num_inducing", "num_inducing_hidden",
        "treatment_lag", "optimize_hyperparams",
        "epochs", "batch_size", "lr", "weight_decay", "seed"
    )
    _explode_for_filtering(row, keys_to_explode, train_cfg, prefix="cfg")

    if metrics:
        # keep full metrics as JSON and surface a couple of common scalars
        row["metrics_json"] = json.dumps(metrics, separators=(",", ":"), ensure_ascii=False)
        for k in ("final_loss", "val_loss", "best_metric"):
            if k in metrics:
                row[k] = metrics[k]
    return row

def find_train(
    root: str | Path,
    model: str | None = None,
    **filters
):
    """
    Quick filter over the index. Example:
       find_train('.', model='seqgplvm', cfg_latent_dim=5, data_run_id='6c39f2a1')
    Returns a pandas DataFrame (may be empty).
    """
    import pandas as pd
    idx_path = Path(root) / "data" / "index" / "training.parquet"
    if not idx_path.exists():
        import pandas as pd
        return pd.DataFrame()
    df = pd.read_parquet(idx_path)
    if model is not None:
        df = df[df["model"] == model]
    for k, v in filters.items():
        if k in df.columns:
            df = df[df[k] == v]
    return df

def upsert_training_index(root: str | Path, row: dict):
    """
    Upsert (model, train_id) into data/index/training.parquet.
    If a row exists for the same keys, replace it; otherwise append.
    """
    import pandas as pd
    idx_path = Path(root) / "results" / "index" / "training.parquet"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])

    if idx_path.exists():
        df = pd.read_parquet(idx_path)
        mask = (df["model"] == row["model"]) & (df["train_id"] == row["train_id"])
        df = df[~mask]
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    # optional: maintain one row per (model, train_id), keep latest
    df.to_parquet(idx_path)

######### Loading Models #########
import re 
_ckpt_re = re.compile(r"step_(\d+)\.pt$")

def latest_checkpoint_path(run_dir: Path) -> Path | None:
    ckpts = list((run_dir / "ckpts").glob("step_*.pt"))
    if not ckpts:
        return None
    def stepnum(p: Path):
        m = _ckpt_re.search(p.name)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=stepnum)

def load_checkpoint(ckpt_path: Path, map_location="cpu"):
    payload = torch.load(ckpt_path, map_location=map_location)
    return payload  # contains model_state, optimizer_state, maybe "extra"

def get_epochs_completed_prior(run_dir: Path) -> int:
    mani = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return int(mani.get("epochs_completed", 0))