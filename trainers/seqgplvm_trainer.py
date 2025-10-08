import hashlib
import torch 
import json 
import numpy as np
import pandas as pd
import gpytorch
from tqdm.auto import trange
from gpytorch.likelihoods import Likelihood
from gpytorch.utils.cholesky import NotPSDError
from pathlib import Path
from models.SeqGPLVM import SeqGPLVM
from utils.inspectors import get_actuals_via_getters 
from utils.preprocessings import get_training_tensors 
from utils.checkpoints import make_train_id, write_train_files, save_ckpt, make_training_index_row, train_dir
from utils.checkpoints import upsert_training_index,latest_checkpoint_path, load_checkpoint, get_epochs_completed_prior
import shutil,os, sys
from utils.pathing import as_path
from utils.progress import ProgressLogger
from typing import  Type 
from utils.training import class_to_id, _update_manifest, tensor_fingerprint
import time, traceback


def train_seqgplvm(df: pd.DataFrame,
                   df_meta_data: dict,
                   latent_dim : int = 1,
                   num_inducing: int = 50, 
                   num_inducing_hidden: int = 5,
                   treatment_lag: int = 1,
                   treatment_model: Type[Likelihood] = None, # "bernoulli" | "gaussian"
                   init_z: torch.Tensor = None, # optional initial Z if none  the model will assign them N(0,1)
                   learn_inducing_locations: bool = True, # whether to optimize the inducing locations or keep them fixed
                   use_titsias: bool = False, # whether to use Titsias' trick for inducing points
                   pid_col: str = "patient_id",
                   time_col: str = "t",
                   treatment_col: str = "D",
                   covariate_cols_prefix: str = "x",
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                   optimize_hyperparams: dict = {"lr": 1e-2, "num_epochs": 20000},
                   checkpoint_interval: int = 2000,
                   param_logging_freq = 50,
                   resume_mode: str = "auto" # "auto" | "yes" | "no"
                   ):

    X,A,id2row = get_training_tensors(df,
                                      id_col=pid_col,
                                      time_col=time_col,
                                      treatment_col=treatment_col,
                                      covariate_cols_prefix=covariate_cols_prefix,
                                      treatment_lag=treatment_lag, 
                                      )
    
    metadata_file_path = df_meta_data["split_file"]
    with open(as_path(metadata_file_path)) as f:
        train_ids = json.load(f)["train_ids"]

    train_rows = [id2row[pid] for pid in train_ids if pid in id2row]
    X_train = X[train_rows].to(device)
    A_train = A[train_rows].to(device)

    # Model:
        
    if treatment_model is None:
        raise ValueError("your treatment model must be specified")
    
    model = SeqGPLVM(Y = A_train, X_cov = X_train, latent_dim = latent_dim, n_inducing_x = num_inducing, n_inducing_hidden = num_inducing_hidden,
                        init_z=init_z, device=device,
                        lik=treatment_model,
                        learn_inducing_locations = learn_inducing_locations,
                        use_titsias=use_titsias).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimize_hyperparams["lr"])
    num_epochs = optimize_hyperparams["num_epochs"]

    # tqdm: pretty (interactive), quiet (batch logs)
    is_tty = sys.stderr.isatty()
    iterator = trange(
        num_epochs,
        leave=is_tty,
        disable=not is_tty,
        dynamic_ncols=True
    )

    if os.environ.get("SLURM_JOB_ID"):
        progress_root = Path(os.environ.get("FINAL_ROOT", "./results")).expanduser()
        plog = ProgressLogger(max_iters=num_epochs, root=progress_root, every=20)
    else:
        class _NoopProgress:
            def update(self, *args, **kwargs): pass
        plog = _NoopProgress()

    # bookkeeping structures

    keywords = ["chol_variational_covar", "variational_mean"] # these two are too big to save so we ommit them during the book keeping
    param_hist = {name: [] for name, _ in model.named_parameters() if not any(kw in name for kw in keywords) }
    actual_params = get_actuals_via_getters(model)
    actual_params = {key: [item] for key,item in actual_params.items()}

    # derive a dataset ID from params 
    data_run_id = df_meta_data.get("run_id") 
    model_name = "seqgplvm"

    final_root = Path(os.environ.get("FINAL_ROOT", "./results")).expanduser()
    final_root.mkdir(parents=True, exist_ok=True)
    scratch_root = final_root


    # build a compact training config dict that determines the training identity
    _train_cfg_identity = {
        "N": X_train.size(0),
        "T": X_train.size(1),
        "C": X_train.size(2),
        "latent_dim": latent_dim,
        "num_inducing": num_inducing,
        "num_inducing_hidden": num_inducing_hidden,
        "treatment_lag": treatment_lag,
        "treatment_model": class_to_id(treatment_model),
        "init_z": tensor_fingerprint(init_z) if init_z is not None else None,
        "learn_inducing_locations": learn_inducing_locations,
        "use_titsias": use_titsias,
        "lr": optimize_hyperparams["lr"]
    }
     
    train_id = make_train_id(
        data_run_id=data_run_id,
        model_name=model_name,
        train_cfg=_train_cfg_identity,
    )

    #project_root = Path(".")
    project_root = scratch_root                     
    train_out = train_dir(project_root, model_name, train_id)

    # Fresh vs resume decision
    resume = False
    if resume_mode == "no":
        if train_out.exists():
            # DANGEROUS: hard reset of this run directory
            shutil.rmtree(train_out)
        # fresh run
    elif resume_mode in ("auto", "yes"):
        if train_out.exists():
            last_ckpt = latest_checkpoint_path(train_out)
            resume = last_ckpt is not None
    else:
        raise ValueError("resume_mode must be one of: 'auto', 'yes', 'no'")



    # data reference for provenance
    data_ref = {
        "dgp": df_meta_data["dgp"],
        "data_file": df_meta_data["path"],      
        "split_file": df_meta_data.get("split_file"),
        "data_run_id": data_run_id,
    }

    if not resume:
        _train_cfg_identity["logging"] = {
        "param_logging_freq": param_logging_freq,
        "checkpoint_interval": checkpoint_interval
        }
        # create output folder and write configs/manifest
        train_out = write_train_files(
            root=scratch_root,
            model_name=model_name,
            train_id=train_id,
            train_cfg=_train_cfg_identity,
            data_ref=data_ref,
        )
    
    loss_list = []

    epochs_completed_prior = 0  # will set properly below
    epochs_completed = 0        # running counter this session
    status = "success"
    error_info = None

    # If resuming, read prior progress from manifest if it exists
    if resume:
        ckpt_path = latest_checkpoint_path(train_out)
        payload = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        extra = payload.get("extra") or {}
        loss_list     = extra.get("loss_list", loss_list)
        param_hist    = extra.get("param_hist", param_hist)
        actual_params = extra.get("actual_params", actual_params)
        epochs_completed_prior = get_epochs_completed_prior(train_out)
        print(f"[resume] {ckpt_path.name} | prior epochs={epochs_completed_prior}")
    
    print(f"\n Training for DGP with paramters: \n {df_meta_data} \n on device {device}")

    try:

        for i in iterator:

            optimizer.zero_grad()
        
            with gpytorch.settings.cholesky_jitter(double_value=1e-3):
                loss = model()
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
            epochs_completed = i + 1
            # Heartbeat (global step = prior + current)
            global_step = epochs_completed + epochs_completed_prior
            try:
                current_lr = optimizer.param_groups[0].get("lr", None)
            except Exception:
                current_lr = None
            plog.update(step=global_step, loss=float(loss.item()), lr=current_lr)

            if i % param_logging_freq == 0:
                for name, p in model.named_parameters():
                    if not any(kw in name for kw in keywords):
                        param_hist[name].append(p.data.clone().detach().cpu().numpy())
                loss_list.append(loss.item())

                real_params = get_actuals_via_getters(model)
                for key,item in real_params.items():
                    actual_params[key].append(item)

            # periodic checkpoint
            if (i + 1) % checkpoint_interval == 0:
                save_ckpt(
                    train_out,
                    step=epochs_completed + epochs_completed_prior,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list}
                )

    except (NotPSDError, RuntimeError) as e:
        if isinstance(e, NotPSDError) or "cholesky" in str(e).lower():
            print(f"🚨 Cholesky/PSD failure at iter {i}: {e}")
            # save right before quitting
            save_ckpt(
                train_out,
                step=epochs_completed + epochs_completed_prior,
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
            "failed_at_global_step": int(epochs_completed + epochs_completed_prior),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            # mark failure in manifest immediately
        _update_manifest(train_out, {"status": status, "error": error_info})
                        # re-raise so you see exactly where it happened:
        raise
    finally:
        # final save if we didn't land exactly on a checkpoint
        if epochs_completed > 0 and (epochs_completed % checkpoint_interval) != 0:
            save_ckpt(
                train_out,
                step=epochs_completed + epochs_completed_prior,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list}
            )
        # finalize progress log
        # final heartbeat (best-effort)
        try:
            plog.update(step=epochs_completed + epochs_completed_prior,
                        loss=(float(loss_list[-1]) if loss_list else None))
        except Exception:
            pass

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
            root=project_root,
            model_name="seqgplvm",
            train_id=train_id,
            train_cfg=_train_cfg_identity,
            data_run_id=data_run_id,
            metrics=metrics,
        )
        upsert_training_index(".", row)



    


    
    





    