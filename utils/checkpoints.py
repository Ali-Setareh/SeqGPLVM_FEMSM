import os 
import glob
from datetime import datetime
from pathlib import Path 
import json, platform, subprocess, gzip
import hashlib
import torch 
import re

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
    return Path(root) / "models" / model_name / train_id

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

def save_ckpt(path: Path,
              step: int,
              model_state: dict,
              optimizer_state: dict | None = None,
              extra: dict | None = None,
              *,
              keep_last: int = 2,
              milestone_every: int = 0,
              compress_older: bool = True) -> Path:
    """
    Save a checkpoint atomically, then prune + (optionally) compress older ones.
    - Keeps the newest .pt uncompressed so latest_checkpoint_path / load_checkpoint continue to work unchanged.
    - Retains 'milestone' steps (multiples of milestone_every) in addition to the last K.
    """
    ckpt_dir = Path(path) / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Atomic write of the new checkpoint
    fname = f"step_{int(step):07d}.pt"
    final_path = ckpt_dir / fname
    tmp_path = final_path.with_suffix(".pt.tmp")
    payload = {"model_state": model_state}
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra:
        payload["extra"] = extra
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)  # atomic rename on same filesystem

    # 2) Prune+compress older checkpoints
    _prune_and_compress_inline(
        ckpt_dir=ckpt_dir,
        newest=final_path,
        keep_last=keep_last,
        milestone_every=milestone_every,
        compress_older=compress_older
    )
    return final_path

_step_re = re.compile(r"^step_(\d+)\.pt(?:\.(?:zst|gz))?$")  # ^ anchor (optional but nice)

def _stepnum(p: Path) -> int:
    m = _step_re.match(p.name)
    return int(m.group(1)) if m else -1

def _prune_and_compress_inline(ckpt_dir: Path,
                               newest: Path,
                               *,
                               keep_last: int = 3,
                               milestone_every: int = 0,
                               compress_older: bool = True) -> None:
    if not ckpt_dir.is_dir():
        return

    # include plain, compressed, and even stray *.pt.tmp so we can clean them
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt*"), key=_stepnum)

    # decide survivor STEPS (newest, last K, milestones)
    newest_step = _stepnum(newest)
    steps_desc = sorted({s for s in (_stepnum(p) for p in all_ckpts) if s >= 0}, reverse=True)

    survivor_steps = set(steps_desc[:keep_last]) if keep_last > 0 else set()
    if milestone_every:
        survivor_steps |= {s for s in steps_desc if s % milestone_every == 0}
    survivor_steps.add(newest_step)

    # delete any file whose step not in survivor_steps (covers .pt, .pt.zst, .pt.gz, .pt.tmp)
    for p in all_ckpts:
        if _stepnum(p) not in survivor_steps:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    if not compress_older:
        return

    # survivors (files) after deletion, for compression decision
    survivors_files = [p for p in (ckpt_dir.glob("step_*.pt*")) if _stepnum(p) in survivor_steps]

    # compress survivors except the newest — only when they are plain .pt
    to_compress = [p for p in survivors_files if p.suffix == ".pt" and _stepnum(p) != newest_step]

    # Prefer zstd; fallback to gzip (pure Python)
    try:
        import zstandard as zstd
        def _compress(p: Path):
            out = p.with_suffix(p.suffix + ".zst")     # .pt.zst
            tmp = out.with_suffix(out.suffix + ".tmp") # .zst.tmp
            c = zstd.ZstdCompressor(level=19)
            with open(p, "rb") as fin, open(tmp, "wb") as fout:
                fout.write(c.compress(fin.read()))
            os.replace(tmp, out)
            p.unlink()
    except Exception:
        import gzip, shutil
        def _compress(p: Path):
            out = p.with_suffix(p.suffix + ".gz")      # .pt.gz
            tmp = out.with_suffix(out.suffix + ".tmp") # .gz.tmp
            with open(p, "rb") as fin, gzip.open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            os.replace(tmp, out)
            p.unlink()

    for p in to_compress:
        try:
            _compress(p)
        except Exception:
            # best-effort; ignore compression errors
            pass

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
        "N", "T", "C",
        "latent_dim", "num_inducing", "num_inducing_hidden",
        "treatment_lag", "treatment_model",
        "init_z", "learn_inducing_locations", "use_titsias",
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

def load_ckpt_any(p: Path, map_location=None):
    p = Path(p)
    suf = p.suffixes[-2:]  # e.g., ['.pt', '.zst'] or ['.pt', '.gz']
    if p.suffix == ".pt":
        return torch.load(p, map_location=map_location)
    if suf == ['.pt', '.zst']:
        import zstandard as zstd
        with open(p, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as rfh:
                return torch.load(rfh, map_location=map_location)
    if suf == ['.pt', '.gz']:
        with gzip.open(p, "rb") as fh:
            return torch.load(fh, map_location=map_location)
    raise ValueError(f"Unrecognized checkpoint format: {p}")

def get_epochs_completed_prior(run_dir: Path) -> int:
    mani = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return int(mani.get("epochs_completed", 0))