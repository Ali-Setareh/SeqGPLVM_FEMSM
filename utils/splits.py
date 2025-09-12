from __future__ import annotations
from pathlib import Path
import json, numpy as np

def split_ids(dgp: str, N: int, split_seed: int = 42, train=0.7, val=0.15, test=0.15):
    assert abs(train + val + test - 1) < 1e-9
    ids = np.arange(1, N + 1)                 # patient IDs assumed 1..N in sims
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(ids)
    n_train = int(round(train * N))
    n_val   = int(round(val   * N))
    train_ids = perm[:n_train].tolist()
    val_ids   = perm[n_train:n_train+n_val].tolist()
    test_ids  = perm[n_train+n_val:].tolist()
    return {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
            "proportions": {"train": train, "val": val, "test": test},
            "split_seed": split_seed, "by": "unit"}

def split_path(dgp: str, N: int, split_seed: int = 42, T: int | None = None, p: int | None = None) -> Path:
    parts = [f"N{N}"]
    if T is not None: parts.append(f"T{T}")
    if p is not None: parts.append(f"p{p}")
    parts.append(f"splitseed{split_seed}.json")
    return Path("data") / "splits" / dgp / "_".join(parts)

def make_or_load_split(dgp: str, N: int, split_seed: int = 42, T: int | None = None, p: int | None = None,
                       train=0.7, val=0.15, test=0.15) -> dict:
    path = split_path(dgp, N, split_seed, T, p)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    sp = split_ids(dgp, N, split_seed, train, val, test)
    path.write_text(json.dumps(sp, indent=2))
    return sp
