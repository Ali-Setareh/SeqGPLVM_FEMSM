from __future__ import annotations
from pathlib import Path
import json, time, glob, os

PROGRESS_DIR = Path(os.environ.get("FINAL_ROOT", ".")) / "progress"

def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def fmt_eta(s):
    if s is None: return "--:--:--"
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        files = sorted(glob.glob(str(PROGRESS_DIR / "task_*_*.json")))
        rows = []
        for f in files:
            d = load(f); 
            if not d: continue
            rows.append((
                int(d.get("task_id") or 0),
                d.get("host",""),
                d.get("step",0),
                d.get("max",1),
                d.get("pct",0.0),
                d.get("loss",None),
                d.get("lr",None),
                d.get("eta_s",None),
            ))
        rows.sort(key=lambda r: r[0])

        os.system("printf '\\033c'")
        print(f"Progress dir: {PROGRESS_DIR}   tasks: {len(rows)}")
        print(f"{'TASK':>4}  {'NODE':<10}  {'STEP':>8}/{ 'MAX':<8}  {'%':>6}  {'LOSS':>12}  {'LR':>10}  {'ETA':>8}")
        for t,node,st,mx,pct,loss,lr,eta_s in rows:
            loss_s = f"{loss:.4f}" if isinstance(loss,(int,float)) else "-"
            lr_s   = f"{lr:.2e}" if isinstance(lr,(int,float)) else "-"
            print(f"{t:>4}  {node:<10}  {st:>8}/{mx:<8}  {pct:>6.2f}  {loss_s:>12}  {lr_s:>10}  {fmt_eta(eta_s):>8}")
        time.sleep(2)

if __name__ == "__main__":
    main()
