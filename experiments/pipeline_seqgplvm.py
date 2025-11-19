from pathlib import Path
import argparse, subprocess, json, os, tempfile

from utils.training import load_train_cfg_from_json, materialize_cfg
from trainers.seqgplvm_propensity import propensity_seqgplvm


def run(cmd_list):
    subprocess.run(cmd_list, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp_config", type=Path, required=True)
    parser.add_argument("--dgp_manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Training config JSON for SeqGPLVM")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0) Get train_id from the *same* training config you already use
    # ------------------------------------------------------------------
    train_cfg = load_train_cfg_from_json(args.config)
    train_cfg = materialize_cfg(train_cfg)
    train_id = train_cfg["train_id"]

    # ------------------------------------------------------------------
    # 1) TRAIN
    # ------------------------------------------------------------------
    print(f"[{train_id}] Starting training…")
    run([
        "python", "-m", "experiments.train_seqgplvm",
        "--dgp_config",   str(args.dgp_config),
        "--dgp_manifest", str(args.dgp_manifest),
        "--config",       str(args.config),
        "--device",       args.device,
    ])
    print(f"[{train_id}] Training done.")

    # ------------------------------------------------------------------
    # 2) VALIDATION
    # ------------------------------------------------------------------
    print(f"[{train_id}] Starting validation…")

    val_cfg = {
        "train_id": train_id,
        "optimize_hyperparams_val": {"lr": 1e-2, "num_epochs": 1000},
        "checkpoint_interval": 200,
        "param_logging_freq": 50,
        "resume_mode": "no",
        "load_data": False,
        "extra_logging": ["loss_list", "param_hist"],
        "extra_logging_mode": "diagnose",
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

    propensity_seqgplvm(
        train_id=train_id,
        sample_count=100,
        load_data=False,
    )

    print(f"[{train_id}] Propensity done.")


if __name__ == "__main__":
    main()
