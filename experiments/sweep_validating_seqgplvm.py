from pathlib import Path
import subprocess, os, json, tempfile
import pandas as pd

INDEX_PATH = Path("results/index/training.parquet")
DEVICE = "auto"

def run(cmd): subprocess.run(cmd, check=True)

def main():
    cfg = {
        "optimize_hyperparams_val": {"lr": 1e-2, "num_epochs": 1000},
        "checkpoint_interval": 200,
        "param_logging_freq": 50,
        "resume_mode": "no",
        "load_data": False, 
        "extra_logging": ["loss_list", "param_hist"],
        "extra_logging_mode": "diagnose"
        }
    
    df = pd.read_parquet(INDEX_PATH)
    train_ids = set(df.loc[df["model"] == "seqgplvm", "train_id"].unique())
    validated_ids = set(df.loc[df["model"] == "seqgplvm_val", "train_id"].unique())
    to_validate = sorted(train_ids - validated_ids)
    #################
    to_validate = ["54ecbe1f75"]  # TEMPORARY LIMIT FOR TESTING
    #################
    if not to_validate:
        print("Nothing to do — all training runs already have seqgplvm_val.")
        return

    is_slurm = "SLURM_ARRAY_TASK_ID" in os.environ
    if is_slurm:
        # Pick exactly one id per array task (1-based)
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
        if not (0 <= idx < len(to_validate)):
            raise SystemExit(f"TASK_ID out of range (size={len(to_validate)})")
        selected = [to_validate[idx]]

        # Unique scratch in TMPDIR to avoid clashes on HPC
        job_id   = os.environ.get("SLURM_JOB_ID", "nojid")
        task_tag = os.environ.get("SLURM_ARRAY_TASK_ID", "single")
        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        scratch  = Path(tmp_root) / f"val_sweep_{job_id}_{task_tag}"
        scratch.mkdir(parents=True, exist_ok=True)
        delete_after = False  # HPC scratch can be auto-cleaned; keep files if you want
    else:
        # Local: use repo configs/ and delete after run
        scratch = Path("configs")
        scratch.mkdir(exist_ok=True)
        selected = to_validate
        delete_after = True

    for tid in selected:
        cfg["train_id"] = tid # the training run to validate
        cfg_path = scratch / f"val_{tid}.json"
        try:
            cfg_path.write_text(json.dumps(cfg))
            run([
                "python", "-m", "experiments.validate_seqgplvm",
                "--config", str(cfg_path),
                "--device", DEVICE,
            ])
        finally:
            if delete_after:
                try: cfg_path.unlink(missing_ok=True)
                except Exception: pass

    
if __name__ == "__main__":
    main()

