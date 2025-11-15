from trainers.seqgplvm_trainer import train_seqgplvm
import argparse, json, yaml, torch, pandas as pd
from pathlib import Path
from utils.runs import load_by_params
from utils.training import load_train_cfg_from_json, materialize_cfg
from dgps import get_simulator

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dgp_config", required=True)
    p.add_argument("--dgp_manifest", required=False)
    p.add_argument("--load_saved_data", required=False, action="store_true")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = p.parse_args()
    
    data_path = Path(args.dgp_config)
    manifest_path = Path(args.dgp_manifest) if args.dgp_manifest else None

    data_cfg = yaml.safe_load(data_path.read_text()) if data_path.suffix in {".yml",".yaml"} else json.loads(data_path.read_text())
    if args.load_saved_data:
        df, manifest = load_by_params(".", data_cfg)
    else:
        if not manifest_path:
            raise ValueError("If --load_saved_data is not set, --dgp_manifest must be provided.")
        
        simulate = get_simulator(data_cfg["dgp"])
        df = simulate(data_cfg)
        manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.suffix in {".yml",".yaml"} else json.loads(manifest_path.read_text())    

    train_cfg = Path(args.config)
    cfg = load_train_cfg_from_json(train_cfg)  # <-- replaces the yaml/json manual load
    cfg = materialize_cfg(cfg, args.device)  # <-- ensures all objects are in place
    
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))

    train_seqgplvm(
        df=df, df_meta_data=manifest, device=device,
        latent_dim=cfg.get("latent_dim", 1),
        num_inducing=cfg.get("num_inducing", 50),
        num_inducing_hidden=cfg.get("num_inducing_hidden", 5),
        treatment_lag=cfg.get("treatment_lag", 1),
        treatment_model = cfg.get("treatment_model", None),
        init_z=cfg.get("init_z", None),
        z_prior=cfg.get("z_prior", "normal"),
        z_initializer=cfg.get("z_initializer", "normal"),
        uniform_halfwidth=cfg.get("uniform_halfwidth", None),
        prior_std=cfg.get("prior_std", None),
        learn_inducing_locations=cfg.get("learn_inducing_locations", True),
        use_titsias=cfg.get("use_titsias", False),
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