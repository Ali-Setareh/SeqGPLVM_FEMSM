import torch
from utils.checkpoints import latest_checkpoint_path, load_checkpoint, lates, load_checkpointt_checkpoint_path, make_train_id, train_dir, train_dir
from utils.preprocessings import get_training_tensors
import pandas as pd
from utils.pathing import as_path
from pathlib import Path
import os
import json
from models.SeqGPLVM import SeqGPLVM, SeqGPLVMVal
from gpytorch.likelihoods import BernoulliLikelihood
import gpytorch
from gpytorch.utils.errors import NotPSDError
import numpy as np
from tqdm import trange
from utils.runs import write_train_files, _update_manifest, load_by_params
import time

def train_seqgplvm_val(df: pd.DataFrame,
                       df_meta_data: dict,
                       latent_dim : int = 1,
                       num_inducing: int = 50, 
                       num_inducing_hidden: int = 5,
                       treatment_lag: int = 1,
                       pid_col: str = "patient_id",
                       time_col: str = "t",
                       treatment_col: str = "D",
                       covariate_cols_prefix: str = "x",
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                       optimize_hyperparams_val: dict = {"lr": 1e-2, "num_epochs": 10000},
                       checkpoint_interval: int = 2000,
                       param_logging_freq: int = 50,
                       resume_mode: str = "auto",
                       parent_resume_mode: str = "auto"   # how to treat the parent train run
                       ):
    """
    Validation fine-tuning: load a trained SeqGPLVM, attach validation latents, 
    freeze all base params, and optimize only Z_val on the validation split.
    """
    # 1) Build *both* splits so we can reconstruct the trained model shape
    X, A, id2row = get_training_tensors(
        df,
        id_col=pid_col, time_col=time_col,
        treatment_col=treatment_col,
        covariate_cols_prefix=covariate_cols_prefix,
        treatment_lag=treatment_lag,
    )

    with open(as_path(df_meta_data["split_file"])) as f:
        split = json.load(f)
    
    # prefer "val_ids" if present; fall back to "test_ids"
    val_ids = split.get("val_ids", [])
    test_ids = split.get("test_ids", [])
    test_val_ids = val_ids + test_ids 
    if len(test_val_ids) == 0:
        raise ValueError("No val_ids or test_ids found in split file.")
    
    val_rows   = [id2row[pid] for pid in test_val_ids   if pid in id2row]

    X_val   = X[val_rows].to(device)
    A_val   = A[val_rows].to(device)

    # 2) Rebuild the *trained* base model with TRAIN shapes and load its weights
    model_base = SeqGPLVM(
        Y=A_train, X_cov=X_train, latent_dim=latent_dim,
        n_inducing_x=num_inducing, n_inducing_hidden=num_inducing_hidden,
        init_z=None, device=device, lik=BernoulliLikelihood,
        learn_inducing_locations=True, use_titsias=False
    ).to(device)

    # Resolve parent train directory and checkpoint
    final_root = Path(os.environ.get("FINAL_ROOT", "./results")).expanduser()
    train_id = make_train_id(
        data_run_id=df_meta_data.get("run_id"),
        model_name="seqgplvm",
        train_cfg={
            "latent_dim": latent_dim,
            "num_inducing": num_inducing,
            "num_inducing_hidden": num_inducing_hidden,
            "treatment_lag": treatment_lag,
            "lr": None,  # lr not needed to reconstruct id; keep None to match training
        },
    )
    parent_out = train_dir(final_root, "seqgplvm", train_id)

    if not parent_out.exists():
        raise FileNotFoundError(f"Parent train run not found: {parent_out}")

    ckpt_path = latest_checkpoint_path(parent_out)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in parent run: {parent_out}")

    payload = load_checkpoint(ckpt_path, map_location=device)
    model_base.load_state_dict(payload["model_state"], strict=True)

    # 3) Wrap the trained model for validation and expose Z_val
    model = SeqGPLVMVal.from_trained(model_base, X_val=X_val, Y_val=A_val).to(device)

    # 4) Freeze everything except the validation latents
    for n, p in model.named_parameters():
        p.requires_grad = False
    # Many repos expose Z_val as a small submodule; if not, fall back to name filter.
    try:
        trainables = list(model.Z_val.parameters())
    except Exception:
        trainables = [p for n, p in model.named_parameters() if "Z_val" in n]
        for p in trainables: p.requires_grad = True

    optimizer = torch.optim.Adam(trainables, lr=optimize_hyperparams_val["lr"])
    num_epochs = optimize_hyperparams_val["num_epochs"]

    # Progress logger (same behavior as train)
    is_tty = sys.stderr.isatty()
    iterator = trange(num_epochs, leave=is_tty, disable=not is_tty, dynamic_ncols=True)

    # progress/heartbeat
    if os.environ.get("SLURM_JOB_ID"):
        plog = ProgressLogger(max_iters=num_epochs,
                              root=final_root, every=20)
    else:
        class _NoopProgress:
            def update(self, *a, **k): pass
        plog = _NoopProgress()

    # 5) Create a *new* run directory for validation, linked to its parent
    val_tag = {"mode": "val", "parent_train_id": train_id}
    val_cfg_identity = {
        "latent_dim": latent_dim,
        "num_inducing": num_inducing,
        "num_inducing_hidden": num_inducing_hidden,
        "treatment_lag": treatment_lag,
        "lr_val": optimize_hyperparams_val["lr"],
        "logging": {"param_logging_freq": param_logging_freq,
                    "checkpoint_interval": checkpoint_interval},
    }

    val_train_id = make_train_id(
        data_run_id=df_meta_data.get("run_id"),
        model_name="seqgplvm_val",
        train_cfg=val_cfg_identity,
    )
    val_out = write_train_files(
        root=final_root,
        model_name="seqgplvm_val",
        train_id=val_train_id,
        train_cfg={**val_cfg_identity, **val_tag},
        data_ref={
            "dgp": df_meta_data["dgp"],
            "data_file": df_meta_data["path"],
            "split_file": df_meta_data.get("split_file"),
            "data_run_id": df_meta_data.get("run_id"),
        },
    )

    # 6) Lightweight bookkeeping (mirror your training function)
    keywords = ["chol_variational_covar", "variational_mean"]
    param_hist = {n: [] for n, p in model.named_parameters() if p.requires_grad and not any(kw in n for kw in keywords)}
    actual_params = get_actuals_via_getters(model); actual_params = {k: [v] for k, v in actual_params.items()}
    loss_list = []
    status, error_info = "success", None

    try:
        for i in iterator:
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(double_value=1e-3):
                loss = model()
            iterator.set_description(f"Loss: {float(np.round(loss.item(), 2))}, iter {i}")
            loss.backward()
            optimizer.step()

            # progress heartbeat
            try:
                plog.update(step=i+1, loss=float(loss.item()),
                            lr=optimizer.param_groups[0].get("lr", None))
            except Exception:
                pass

            if i % param_logging_freq == 0:
                for n, p in model.named_parameters():
                    if p.requires_grad and not any(kw in n for kw in keywords):
                        param_hist[n].append(p.data.detach().cpu().numpy())
                loss_list.append(loss.item())
                real_params = get_actuals_via_getters(model)
                for k, v in real_params.items():
                    actual_params[k].append(v)

            if (i + 1) % checkpoint_interval == 0:
                save_ckpt(
                    val_out,
                    step=i+1,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list}
                )

    except (NotPSDError, RuntimeError) as e:
        if isinstance(e, NotPSDError) or "cholesky" in str(e).lower():
            print(f"🚨 Cholesky/PSD failure at iter {i}: {e}")
            save_ckpt(
                val_out,
                step=i+1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list}
            )
        status = "failed"
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "failed_at_iter": int(i),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _update_manifest(val_out, {"status": status, "error": error_info})
        raise
    finally:
        if (len(loss_list) == 0) or ((i + 1) % checkpoint_interval != 0):
            save_ckpt(
                val_out,
                step=i+1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list}
            )
        try:
            plog.update(step=i+1, loss=(float(loss_list[-1]) if loss_list else None))
        except Exception:
            pass
        _update_manifest(val_out, {
            "status": status,
            "epochs_completed": int(i+1),
            "parent_train_dir": str(parent_out),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

