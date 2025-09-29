from __future__ import annotations
from utils.pathing import as_path
from pathlib import Path
import json, os, time
from dataclasses import dataclass, asdict


@dataclass
class ProgressPayload:
    job_id: str | None
    task_id: str | None
    host: str
    step: int
    max: int
    pct: float
    loss: float | None
    lr: float | None
    eta_s: int | None
    ts: int

class ProgressLogger:
    """
    Writes a small JSON 'heartbeat' file per array task.
    Example path: <root>/progress/task_<JOB>_<TASK>.json
    """
    def __init__(self, max_iters: int, root: Path | str, every: int = 100):
        self.max_iters = int(max_iters)
        self.every = int(max(1, every))
        root = as_path(root)
        root.mkdir(parents=True, exist_ok=True)
        jid = os.environ.get("SLURM_JOB_ID", "nojid")
        tid = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        self.path = root / "progress" / f"task_{jid}_{tid}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_step = -10**9
        self._t0 = time.time()

    def update(self, step: int, loss: float | None = None, lr: float | None = None):
        if (step - self._last_step) < self.every and step < self.max_iters:
            return
        self._last_step = step
        now = time.time()
        pct = 100.0 * step / max(1, self.max_iters)
        # ETA (seconds), guarded
        rate = (step / (now - self._t0)) if (now > self._t0) else 0.0
        eta_s = int((self.max_iters - step) / rate) if rate > 1e-8 else None

        payload = ProgressPayload(
            job_id=os.environ.get("SLURM_JOB_ID"),
            task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            host=os.uname().nodename if hasattr(os, "uname") else "",
            step=int(step),
            max=int(self.max_iters),
            pct=round(pct, 2),
            loss=(None if loss is None else float(loss)),
            lr=(None if lr is None else float(lr)),
            eta_s=eta_s,
            ts=int(now),
        )
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(payload)))
        tmp.replace(self.path)  # atomic-ish
