from trainers.seqgplvm_trainer import train_seqgplvm
import argparse, json, yaml, torch, pandas as pd
from pathlib import Path
from utils.runs import load_by_params

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = p.parse_args()
    
    data_path = Path(args.data)

    data_cfg = yaml.safe_load(data_path.data.read_text()) if data_path.suffix in {".yml",".yaml"} else json.loads(data_path.read_text())
    df, manifest = load_by_params(".", data_cfg)

    train_cfg = Path(args.config)
    cfg = yaml.safe_load(train_cfg.read_text()) if train_cfg.suffix in {".yml",".yaml"} else json.loads(train_cfg.read_text())

    
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))

    train_seqgplvm(
        df=df, df_meta_data=manifest, device=device,
        latent_dim=cfg.get("latent_dim", 1),
        num_inducing=cfg.get("num_inducing", 50),
        num_inducing_hidden=cfg.get("num_inducing_hidden", 5),
        treatment_lag=cfg.get("treatment_lag", 1),
        pid_col=cfg.get("pid_col", "patient_id"),
        time_col=cfg.get("time_col", "t"),
        treatment_col=cfg.get("treatment_col", "D"),
        covariate_cols_prefix=cfg.get("covariate_cols_prefix", "x"),
        optimize_hyperparams=cfg.get("optimize_hyperparams", {"lr":1e-2,"num_epochs":20000}),
        checkpoint_interval=cfg.get("checkpoint_interval", 2000),
        param_logging_freq=cfg.get("param_logging_freq", 50),
        resume_mode=cfg.get("resume_mode", "auto")
    )

if __name__ == "__main__":
    main()