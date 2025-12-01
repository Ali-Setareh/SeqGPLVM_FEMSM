from pathlib import Path
import argparse, subprocess, json, os, tempfile

import yaml

from dgps import get_simulator
from utils.pathing import as_path
from utils.training import load_train_cfg_from_json, materialize_cfg
from trainers.seqgplvm_propensity import propensity_seqgplvm
import numpy as np
import pandas as pd

#from rpy2.robjects import r, default_converter
#from rpy2.robjects.conversion import localconverter
#from rpy2.robjects import pandas2ri
#from rpy2 import rinterface
#import trainers.seqgplvm_msm_r  # ensure R function is defined

from trainers.seqgplvm_msm_py import seqgplvm_msm_from_py_py


def run(cmd_list):
    subprocess.run(cmd_list, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp_config", type=Path, required=True)
    parser.add_argument("--dgp_manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Training config JSON for SeqGPLVM")
    parser.add_argument("--train_cfg_identity", type=Path, required=True,
                        help="Training config identity JSON for SeqGPLVM")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0) Get train_id from the *same* training config you already use
    # ------------------------------------------------------------------
    train_cfg = load_train_cfg_from_json(args.config)
    train_cfg = materialize_cfg(train_cfg, device=args.device)
    train_id = train_cfg["train_id"]
    drop_monotone = train_cfg.get("drop_monotone", False)
    dgp_index_path = train_cfg.get("dgp_index_path", None)

    # Where to save MSM results
    final_root = Path(os.environ.get("FINAL_ROOT", "./results")).expanduser()
    msm_dir = final_root / "msm" / "seqgplvm"
    msm_dir.mkdir(parents=True, exist_ok=True)
    

    # ------------------------------------------------------------------
    # 1) TRAIN
    # ------------------------------------------------------------------
    print(f"[{train_id}] Starting training…")
    run([
        "python", "-m", "experiments.train_seqgplvm",
        "--dgp_config",   str(args.dgp_config),
        "--dgp_manifest", str(args.dgp_manifest),
        "--config",       str(args.config),
        "--train_cfg_identity", str(args.train_cfg_identity),
        "--device",       args.device
    ])
    print(f"[{train_id}] Training done.")

    # ------------------------------------------------------------------
    # 2) VALIDATION
    # ------------------------------------------------------------------
    print(f"[{train_id}] Starting validation…")

    val_cfg = {
        "train_id": train_id,
        "optimize_hyperparams_val": {"lr": 1e-2, "num_epochs": 100},
        "checkpoint_interval": 2000,
        "param_logging_freq": 50,
        "resume_mode": "no",
        "load_data": False,
        "extra_logging": ["loss_list", "param_hist"],
        "extra_logging_mode": "experiment",
        "drop_monotone": drop_monotone, # whether to drop monotone rows during validation same as training
        "dgp_index_path": dgp_index_path
    }

    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    val_cfg_path = tmp_root / f"seqgplvm_val_{train_id}.json"
    val_cfg_path.write_text(json.dumps(val_cfg))

    try:
        run([
            "python", "-m", "experiments.validate_seqgplvm",
            "--config", str(val_cfg_path),
            "--device", args.device,
        ])
    finally:
        # Locally you might want to clean; on Helix TMPDIR is auto-cleaned anyway
        try:
            val_cfg_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"[{train_id}] Validation done.")

    # ------------------------------------------------------------------
    # 3) PROPENSITY
    # ------------------------------------------------------------------
    print(f"[{train_id}] Computing propensity…")

    propensity_scores  = propensity_seqgplvm(
        train_id=train_id,
        sample_count=100,
        load_data=False,
        save_propensity=False,
        drop_monotone=drop_monotone,
        dgp_index_path=dgp_index_path
    )

    print(f"[{train_id}] Propensity done.")

    # ------------------------------------------------------------------
    # 4) MSM ESTIMATION
    # ------------------------------------------------------------------
    data_path = Path(args.dgp_config)
    manifest_path = Path(args.dgp_manifest) if args.dgp_manifest else None
    data_cfg = yaml.safe_load(data_path.read_text()) if data_path.suffix in {".yml",".yaml"} else json.loads(data_path.read_text())
    simulate = get_simulator(data_cfg["dgp"])
    df = simulate(data_cfg)
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.suffix in {".yml",".yaml"} else json.loads(manifest_path.read_text())
    print(f"[{train_id}] Starting MSM estimation…")

    P, T, S = propensity_scores.shape
    arr = propensity_scores.detach().cpu().numpy().reshape(P*T, S)

    # id columns
    patient_id = np.repeat(np.arange(P), T)
    t = np.tile(np.arange(1,T+1), P)

    # build dataframe
    batch_cols = [f"phat_batch_{i+1}" for i in range(S)]
    propensity_df = pd.DataFrame(arr, columns=batch_cols)
    propensity_df.insert(0, "t", t)
    propensity_df.insert(0, "patient_id", patient_id)

    df_phat = df.merge(propensity_df,on=["patient_id", "t"], how="inner")

    df_phat["phat_mean"] = df_phat[batch_cols].mean(axis=1)
    df_phat["phat_std"] = df_phat[batch_cols].std(axis=1)
    df_phat["train_id"] = train_id

    

    k_last = manifest["params"].get("max_lag_d", 4)+1
    a_val = manifest["params"].get("a", None)
    data_id = manifest.get("run_id", None)
    with open(as_path(manifest.get("split_file"))) as f:
        splits = json.load(f)
    train_ids = splits["train_ids"]
    val_ids = splits["val_ids"] + splits["test_ids"]
    
    x_cols = [col for col in df_phat.columns if col.startswith("x")]
    
    res_train_py = seqgplvm_msm_from_py_py(df_phat, train_ids, batch_cols, k_last, a_val, data_id, x_cols)
    res_test_py = seqgplvm_msm_from_py_py(df_phat, val_ids, batch_cols, k_last, a_val, data_id, x_cols)
            
        
    # Tag which subset this is (train vs val/test)
    res_train_py["subset"] = "train"
    res_test_py["subset"]  = "val"   # or "test" / "val+test", up to you
    res_train_py["train_id"] = train_id
    res_test_py["train_id"]  = train_id
    # Also store whether this run dropped monotone units at the model stage
    res_train_py["drop_monotone_model"] = drop_monotone
    res_test_py["drop_monotone_model"]  = drop_monotone

    

    if not res_train_py.empty and not res_test_py.empty:
        msm_df = pd.concat([res_train_py, res_test_py], ignore_index=True)

        out_path = msm_dir / f"{train_id}_msm.parquet"
        msm_df.to_parquet(out_path, index=False)
        print(f"[{train_id}] Saved MSM results to {out_path} (n={len(msm_df)})")
    else:
        print(f"[{train_id}] No MSM results to save.")
        

       


if __name__ == "__main__":
    main()
