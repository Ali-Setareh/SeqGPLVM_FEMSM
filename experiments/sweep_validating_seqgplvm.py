# sweep_validate.py
from pathlib import Path
import subprocess
import json
import pandas as pd

INDEX_PATH = Path("results/index.parquet")
CFG_DIR = Path("configs")

def run(cmd_list): subprocess.run(cmd_list, check=True)

def main():
    df = pd.read_parquet(INDEX_PATH)

    train_ids = set(df.loc[df["model"] == "seqgplvm", "train_id"].unique())
    validated_ids = set(df.loc[df["model"] == "seqgplvm_val", "train_id"].unique())
    to_validate = sorted(train_ids - validated_ids)

    if not to_validate:
        print("✅ Nothing to do — all training runs already have seqgplvm_val.")
        return

    CFG_DIR.mkdir(parents=True, exist_ok=True)

    for tid in to_validate:
        cfg_path = CFG_DIR / f"val_{tid}.json"
        cfg_path.write_text(json.dumps({"train_id": tid, 
                                        "optimize_hyperparams": {"lr":1e-2,"num_epochs":10000},
                                        "resume_mode": "no"}))  

        run(["python", "-m", "experiments.train_seqgplvm_val", "--config", str(cfg_path)])

    print(f"🎯 Launched validations for {len(to_validate)} train_id(s).")

if __name__ == "__main__":
    main()
