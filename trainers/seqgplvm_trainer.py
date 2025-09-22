import torch 
import json 
import os
import numpy as np
import pandas as pd
import shutil
import gpytorch
from tqdm.auto import trange
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.utils.cholesky import NotPSDError
from pathlib import Path
from models.SeqGPLVM import SeqGPLVM
from utils.checkpoints import save_checkpoint 
from utils.inspectors import get_actuals_via_getters 
from utils.preprocessings import get_training_tensors 
from utils.checkpoints import make_train_id, write_train_files, save_ckpt

def train_seqgplvm(df: pd.DataFrame,
                   df_meta_data: dict,
                   latent_dim : int = 1,
                   num_inducing: int = 50, 
                   num_inducing_hidden: int =5,
                   treatment_lag: int =1,
                   pid_col: str = "patient_id",
                   time_col: str = "t",
                   treatment_col: str = "D",
                   covariate_cols_prefix: str = "x",
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                   optimize_hyperparams: dict = {"lr": 1e-2, "num_epochs": 20000},
                   checkpoint_interval: int = 2000,
                   param_logging_freq = 50,
                   split_folder: str = "data/splits",
                   checkpoint_folder: str = "results/logs"
                   ):

    X,A,id2row = get_training_tensors(df,
                                      id_col=pid_col,
                                      time_col=time_col,
                                      treatment_col=treatment_col,
                                      covariate_cols_prefix=covariate_cols_prefix,
                                      treatment_lag=treatment_lag, 
                                      )
    
    metadata_file_path = df_meta_data["split_file"]
    with open(Path(metadata_file_path)) as f:
        train_ids = json.load(f)["train_ids"]

    train_rows = [id2row[pid] for pid in train_ids if pid in id2row]
    X_train = X[train_rows].to(device)
    A_train = A[train_rows].to(device)

    # Modol:
    if df_meta_data["params"]["treatment_model"] in ["logit", "probit"]:
        #init_z = torch.nn.Parameter(torch.zeros(A_train.shape[0], latent_dim))
        init_z = None # the model will assign them N(0,1)

        model = SeqGPLVM(Y = A_train, X_cov = X_train, latent_dim = latent_dim, n_inducing_x = num_inducing, n_inducing_hidden = num_inducing_hidden,
                        init_z=init_z, device=device,
                        lik=BernoulliLikelihood,
                        learn_inducing_locations = True,
                        use_titsias=False).to(device)

    else:
        raise ValueError("Only binary model is available")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimize_hyperparams["lr"])
    num_epochs = optimize_hyperparams["num_epochs"]
    
    iterator = trange(num_epochs, leave=True)

    keywords = ["chol_variational_covar", "variational_mean"] # these two are too big to save so we ommit them during the book keeping
    param_hist = {name: [] for name, _ in model.named_parameters() if not any(kw in name for kw in keywords) }
    actual_params = get_actuals_via_getters(model)
    actual_params = {key: [item] for key,item in actual_params.items()}


    print(f"\n Training for DGP with paramters: \n {df_meta_data} \n on device {device}")

    #data_file_path = df_meta_data["path"]
    #base = Path(data_file_path).stem
    

    #ckpt_dir = Path(checkpoint_folder)/ df_meta_data["dgp"] /f"{base}"

    # derive a dataset ID from params 
    params = df_meta_data["params"]
    data_run_id = df_meta_data.get("run_id") 
    model_name = "seqgplvm"

    # build a compact training config dict that determines the training identity
    _train_cfg_identity = {
        "latent_dim": latent_dim,
        "num_inducing": num_inducing,
        "num_inducing_hidden": num_inducing_hidden,
        "treatment_lag": treatment_lag,
        "optimize_hyperparams": optimize_hyperparams,
        "pid_col": pid_col, "time_col": time_col, "treatment_col": treatment_col,
        "covariate_cols_prefix": covariate_cols_prefix,
    }

    train_id = make_train_id(
        data_run_id=data_run_id,
        model_name=model_name,
        train_cfg=_train_cfg_identity,
    )

    # data reference for provenance
    data_ref = {
        "dgp": df_meta_data["dgp"],
        "data_file": df_meta_data["path"],      
        "split_file": df_meta_data.get("split_file"),
        "data_run_id": data_run_id,
    }

    # create output folder and write configs/manifest
    train_out = write_train_files(
        root=Path("."),
        model_name=model_name,
        train_id=train_id,
        train_cfg=_train_cfg_identity,
        data_ref=data_ref,
    )

    loss_list = []

    for i in iterator:

        optimizer.zero_grad()
        try:
            with gpytorch.settings.cholesky_jitter(double_value=1e-3):
              loss = model()
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
            if (i+1) %param_logging_freq == 0:
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
                    step=i+1,
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
                    step=i+1,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list}
                )
            # re-raise so you see exactly where it happened:
            raise
        finally:
            if loss_list:
                iterator.set_description(f"Loss: {loss_list[-1]:.4f}, iter {i}")

    # final save
    if (i + 1) % checkpoint_interval != 0:        
        save_ckpt(
                    train_out,
                    step=i+1,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    extra={'param_hist': param_hist, 'actual_params': actual_params, 'loss_list': loss_list}
                )


    


    
    





    