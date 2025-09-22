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

'''
def save_checkpoint(
    model, 
    optimizer=None, 
    hist_dict=None, 
    step=None, 
    base=None, 
    dir_path=None,
    dataset_meta: dict | None = None,   
):
    # Base checkpoint with model state
    ckpt = {
        'model_state': model.state_dict()
    }

    # Add optional fields if provided
    if step is not None:
        ckpt['step'] = step
    if optimizer is not None:
        ckpt['opt_state'] = optimizer.state_dict()
        ckpt['optimizer_kwargs'] = getattr(optimizer, 'defaults', None)
    if hist_dict is not None:
        ckpt['history'] = hist_dict
    if dataset_meta is not None:
        ckpt['dataset_meta'] = dataset_meta
    if hasattr(model, 'config'):
        ckpt['model_kwargs'] = model.config
    if hasattr(model, "val_meta") and model.val_meta is not None:
        ckpt["model_kwargs"] = dict(ckpt["model_kwargs"], val_meta=model.val_meta)

    # Save to file only if directory is given
    if dir_path is not None:
        os.makedirs(dir_path, exist_ok=True)
        pattern = os.path.join(dir_path, "*.pt")
        for old in glob.glob(pattern):
            try:
                os.remove(old)
            except OSError:
                pass
        filename_parts = ["seqgplvm"]
        if base is not None:
            filename_parts.append(str(base))
        if step is not None:
            filename_parts.append(str(step))
        filename = "_".join(filename_parts) + ".pt"
        torch.save(ckpt, os.path.join(dir_path, filename))
        print(f"\r[Saved checkpoint at step {step if step is not None else 'N/A'}]", end="")
    else:
        print("[Checkpoint created but not saved to disk]")

        return ckpt


def load_checkpoint(checkpoint_path, model_class, optimizer_class, device=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 1) re-create model & optimizer from saved kwargs
    model_kwargs     = ckpt['model_kwargs']
    optimizer_kwargs = ckpt['optimizer_kwargs']

    model     = model_class(**model_kwargs).to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # 2) load their states
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['opt_state'])

    # 3) unpack the rest
    history     = ckpt.get('history', {})
    start_epoch = ckpt.get('step', 0)
    #best_state  = ckpt.get('best_state', None)

    return model, optimizer, history, start_epoch

import torch

def load_checkpoint(
    checkpoint_path,
    model_class=None,
    optimizer_class=None,
    *,
    model=None,
    optimizer=None,
    device=None,
    recreate_model = True
):
    """
    Load a checkpoint with maximal flexibility.

    You can either pass `model`/`optimizer` instances to load into,
    or pass `model_class`/`optimizer_class` to (re)construct them from
    kwargs stored in the checkpoint (if present).

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt file.
    model_class : type, optional
        Class to construct a model from if `model` is not provided.
    optimizer_class : type, optional
        Class to construct an optimizer from if `optimizer` is not provided.
    model : nn.Module, optional
        Existing model instance to load weights into.
    optimizer : torch.optim.Optimizer, optional
        Existing optimizer instance to load state into.
    device : torch.device or str, optional
        Map checkpoint to this device. Defaults to CPU if None.
    strict : bool
        Forwarded to `model.load_state_dict(strict=...)`.

    Returns
    -------
    model, optimizer, history, start_step
    """
    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if recreate_model: 

        # --- Re/Create model ---
        model_kwargs = ckpt.get('model_kwargs', None)

        if model is None:
            if model_class is None:
                raise ValueError(
                    "No `model` instance provided and `model_class` is None. "
                    "Provide a model instance or a model class to reconstruct from checkpoint."
                )
            if model_kwargs is not None:
                model = model_class(**model_kwargs)
            else:
                # Fall back to no-arg construction if kwargs weren’t saved.
                model = model_class()
        model = model.to(device)

        # Load model state if available
        model_state = ckpt.get('model_state', None)
        if model_state is not None:
            model.load_state_dict(model_state, strict = False)#, strict=strict)

    # --- Re/Create optimizer (optional) ---
    optimizer_kwargs = ckpt.get('optimizer_kwargs', None)

    if optimizer is None and optimizer_class is not None:
        if optimizer_kwargs is not None:
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        else:
            optimizer = optimizer_class(model.parameters())

    # Load optimizer state if we have both the state and an optimizer object
    opt_state = ckpt.get('opt_state', None)
    if optimizer is not None and opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except ValueError:
            # Shape or param-group mismatch—continue without raising
            # (often happens when architecture changed slightly).
            pass

    # --- Optional extras ---
    history = ckpt.get('history', {}) or {}
    start_step = ckpt.get('step', 0) or 0

    return model, optimizer, history, start_step

def format_param_key_val(key, val):
    key_str = ''.join(word.title() for word in key.split('_'))
    if isinstance(val, bool):
        val_str = str(int(val))
    else:
        val_str = str(val).replace('-', '-').replace('.', 'p')
    return f"{key_str}{val_str}"
'''
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
    idx_path = Path(root) / "data" / "index" / "training.parquet"
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