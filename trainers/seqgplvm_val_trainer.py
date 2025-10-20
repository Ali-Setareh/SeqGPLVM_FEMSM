import shutil
import torch
from utils.checkpoints import latest_checkpoint_path, load_checkpoint, load_ckpt_any,train_dir, train_dir, get_epochs_completed_prior, upsert_training_index, make_training_index_row
from utils.preprocessings import get_training_tensors
import pandas as pd
from utils.pathing import as_path
from pathlib import Path
from models.SeqGPLVM import SeqGPLVM, SeqGPLVMVal
import gpytorch
from gpytorch.utils.errors import NotPSDError
import numpy as np
from tqdm import trange
from utils.checkpoints import write_train_files

from utils.training import _update_manifest, materialize_cfg, load_train_cfg_from_json
from utils.inspectors import get_actuals_via_getters
import time, sys, os, traceback, json 
from utils.checkpoints import save_ckpt 


def train_seqgplvm_val(train_id: str,
                       pid_col: str = "patient_id",
                       time_col: str = "t",
                       treatment_col: str = "D",
                       covariate_cols_prefix: str = "x",
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                       optimize_hyperparams_val: dict = {"lr": 1e-2, "num_epochs": 10000},
                       checkpoint_interval: int = 2000,
                       param_logging_freq: int = 50,
                       resume_mode: str = "auto",
                       ):
    """
    Validation fine-tuning: load a trained SeqGPLVM, attach validation latents, 
    freeze all base params, and optimize only Z_val on the validation split.
    """
    # 1) Build *both* splits so we can reconstruct the trained model shape
    final_root = Path(os.environ.get("FINAL_ROOT", "./results")).expanduser()
    train_out = train_dir(final_root, "seqgplvm", train_id)
    if not train_out.exists():
        raise FileNotFoundError(f"Parent train run not found: {train_out}")
    
    data_ref  = json.loads((train_out / "data_ref.json").read_text(encoding="utf-8"))
    train_conf = load_train_cfg_from_json((train_out / "config.json"))
    train_conf = materialize_cfg(train_conf, device)
    

    df = pd.read_parquet(as_path(data_ref["data_file"]) / "data.parquet")
    df_manifest = json.loads((as_path(data_ref["data_file"]) / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads(as_path(data_ref["split_file"]).read_text(encoding="utf-8"))

    X, A, id2row = get_training_tensors(
        df,
        id_col=pid_col, time_col=time_col,
        treatment_col=treatment_col,
        covariate_cols_prefix=covariate_cols_prefix,
        treatment_lag=train_conf["treatment_lag"],
    )
    
    # prefer "val_ids" if present; fall back to "test_ids"
    train_ids = split.get("train_ids", [])
    train_rows = [id2row[pid] for pid in train_ids if pid in id2row] 
    X_train = X[train_rows].to(device)
    A_train = A[train_rows].to(device)

    val_ids = split.get("val_ids", [])
    test_ids = split.get("test_ids", [])
    test_val_ids = val_ids + test_ids 
    if len(test_val_ids) == 0:
        raise ValueError("No val_ids or test_ids found in split file.")
    
    val_rows   = [id2row[pid] for pid in test_val_ids   if pid in id2row]

    X_val   = X[val_rows].to(device)
    A_val   = A[val_rows].to(device)

    # 2) Rebuild the *trained* base model with TRAIN shapes and load its weights
    model_base = SeqGPLVM(Y = A_train, X_cov = X_train, latent_dim = train_conf["latent_dim"], 
                     n_inducing_x = train_conf["num_inducing"], n_inducing_hidden = train_conf["num_inducing_hidden"],
                     init_z=None,z_initializer= train_conf["z_initializer"],
                     uniform_halfwidth=train_conf.get("uniform_halfwidth", None),prior_std=train_conf.get("prior_std", None),device=device,
                     lik= train_conf["treatment_model"],
                     learn_inducing_locations = train_conf["learn_inducing_locations"],
                     use_titsias=train_conf["use_titsias"]).to(device)


    ckpt_path = latest_checkpoint_path(train_out)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in parent run: {train_out}")

    payload = load_ckpt_any(ckpt_path, map_location=device)
    model_base.load_state_dict(payload["model_state"], strict=True)

    # 3) Wrap the trained model for validation and expose Z_val
    model = SeqGPLVMVal.from_trained(model_base, X_val=X_val, Y_val=A_val).to(device)

    model.train()    # turn on training mode for ELBO/lik
    for lik in model.likelihoods: lik.train()

    optimizer = torch.optim.Adam(model.Z_val.parameters(), lr=optimize_hyperparams_val["lr"])
    num_epochs = optimize_hyperparams_val["num_epochs"]

    # Progress logger (same behavior as train)
    is_tty = sys.stderr.isatty()
    iterator = trange(num_epochs, leave=is_tty, disable=not is_tty, dynamic_ncols=True)

    # 5) Create a *new* run directory for validation, linked to its parent
    val_cfg_identity = {
        "mode": "val", 
        "parent_train_id": train_id,
        "lr_val": optimize_hyperparams_val["lr"],
    }
    
    model_name = "seqgplvm_val"
    val_out = train_dir(final_root, model_name, train_id)

    # Fresh vs resume decision
    resume = False
    if resume_mode == "no":
        if val_out.exists():
            # DANGEROUS: hard reset of this run directory
            shutil.rmtree(val_out)
        # fresh run
    elif resume_mode in ("auto", "yes"):
        if val_out.exists():
            last_ckpt = latest_checkpoint_path(val_out)
            resume = last_ckpt is not None
    else:
        raise ValueError("resume_mode must be one of: 'auto', 'yes', 'no'")
    
    if not resume:
        val_cfg_identity["logging"] = {"param_logging_freq": param_logging_freq,
                                         "checkpoint_interval": checkpoint_interval}

        val_out = write_train_files(
            root = final_root,
            model_name = model_name,
            train_id=train_id,
            train_cfg=val_cfg_identity,
            data_ref={
                "dgp": df_manifest["dgp"],
                "data_file": df_manifest["path"],
                "split_file": df_manifest.get("split_file"),
                "data_run_id": df_manifest.get("run_id"),
            },
        )

    # 6) Lightweight bookkeeping (mirror your training function)
    keywords = ["chol_variational_covar", "variational_mean"]
    param_hist = {n: [] for n, p in model.named_parameters() if p.requires_grad and not any(kw in n for kw in keywords)}
    actual_params = get_actuals_via_getters(model); actual_params = {k: [v] for k, v in actual_params.items()}
    loss_list = []
    status, error_info = "success", None
    epochs_completed_prior = 0  # will set properly below
    epochs_completed = 0        # running counter this session

    # If resuming, read prior progress from manifest if it exists
    if resume:
        ckpt_path = latest_checkpoint_path(val_out)
        payload = load_ckpt_any(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        extra = payload.get("extra") or {}
        loss_list     = extra.get("loss_list", loss_list)
        param_hist    = extra.get("param_hist", param_hist)
        actual_params = extra.get("actual_params", actual_params)
        epochs_completed_prior = get_epochs_completed_prior(val_out)
        print(f"[resume] {ckpt_path.name} | prior epochs={epochs_completed_prior}")
    
    print(f"\n Validation training for DGP with paramters: \n {df_manifest} \n on device {device}")

    try:
        for i in iterator:
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(double_value=1e-3):
                loss = model()
            iterator.set_description(f"Loss: {float(np.round(loss.item(), 2))}, iter {i}")
            loss.backward()
            optimizer.step()
            epochs_completed = i + 1
            # Heartbeat (global step = prior + current)
            global_step = epochs_completed + epochs_completed_prior

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
                    step=epochs_completed + epochs_completed_prior,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    #extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list},
                    keep_last=1,
                    milestone_every=0
                )

    except (NotPSDError, RuntimeError) as e:
        if isinstance(e, NotPSDError) or "cholesky" in str(e).lower():
            print(f"🚨 Cholesky/PSD failure at iter {i}: {e}")
            save_ckpt(
                val_out,
                step=epochs_completed + epochs_completed_prior,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list},
                keep_last=1, 
                milestone_every=0
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
        #if (len(loss_list) == 0) or ((i + 1) % checkpoint_interval != 0):
        save_ckpt(
                val_out,
                step=epochs_completed + epochs_completed_prior,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list},
                keep_last=1, 
                milestone_every=0
            )
        
        _update_manifest(val_out, {
            "status": status,
            "epochs_completed": int(i+1),
            "parent_train_dir": str(train_out),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        # update manifest with progress & status (success path will overwrite failed → success if no exception)
        mani_patch = {
            "status": status,
            "epochs_completed": int(epochs_completed_prior + epochs_completed),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if error_info:
            mani_patch["error"] = error_info
        _update_manifest(train_out, mani_patch)

        final_epochs = int(epochs_completed_prior + epochs_completed)
        final_loss = float(loss_list[-1]) if loss_list else None

        metrics = {
            "final_loss": final_loss,
            "epochs_completed": final_epochs,
            "status": status,
        }

        row = make_training_index_row(
            root=final_root,
            model_name=model_name,
            train_id=train_id,
            train_cfg=val_cfg_identity,
            data_run_id=df_manifest.get("run_id"),
            metrics=metrics,
        )
        upsert_training_index(".", row)