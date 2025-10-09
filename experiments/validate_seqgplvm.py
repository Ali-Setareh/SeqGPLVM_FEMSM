from trainers.seqgplvm_val_trainer import train_seqgplvm_val
import argparse, json, yaml, torch, pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = p.parse_args()
    
    train_cfg = Path(args.config)
    cfg = yaml.safe_load(train_cfg.read_text()) if train_cfg.suffix in {".yml",".yaml"} else json.loads(train_cfg.read_text())

    
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))

    train_seqgplvm_val(
        device=cfg.get("device", device),
        train_id=cfg.get("train_id", None),
        pid_col=cfg.get("pid_col", "patient_id"),
        time_col=cfg.get("time_col", "t"),
        treatment_col=cfg.get("treatment_col", "D"),
        covariate_cols_prefix=cfg.get("covariate_cols_prefix", "x"),
        optimize_hyperparams=cfg.get("optimize_hyperparams", {"lr":1e-2,"num_epochs":10000}),
        checkpoint_interval=cfg.get("checkpoint_interval", 2000),
        param_logging_freq=cfg.get("param_logging_freq", 50),
        resume_mode=cfg.get("resume_mode", "auto")
    )

if __name__ == "__main__":
    main()