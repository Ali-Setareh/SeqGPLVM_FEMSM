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
    with open(Path(split_folder)/metadata_file_path) as f:
        train_ids = json.load(f)["train_ids"]

    train_rows = [id2row[pid] for pid in train_ids if pid in id2row]
    X_train = X[train_rows].to(device)
    A_train = A[train_rows].to(device)

    # Modol:
    if df_meta_data["treatment_likelihood"] in ["logit", "probit"]:
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


    print(f"Training for DGP with paramters: \n {item[1]}")

    data_file_name = df_meta_data["data_file"]
    base, _ = os.path.splitext(data_file_name)

    ckpt_dir = checkpoint_folder/ df_meta_data["dgp"] /f"{base}"
    # remove the already existing directory
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    os.makedirs(ckpt_dir,exist_ok=True)

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
                save_checkpoint(model, optimizer,
                                {'param_hist': param_hist,
                                'actual_params': actual_params,
                                'loss_list': loss_list},
                                step=i+1,
                                base=base,
                                dir_path=ckpt_dir)
        except (NotPSDError, RuntimeError) as e:
            if isinstance(e, NotPSDError) or "cholesky" in str(e).lower():
                print(f"🚨 Cholesky/PSD failure at iter {i}: {e}")
                # save right before quitting
                save_checkpoint(model, optimizer,
                                {'param_hist': param_hist,
                                'actual_params': actual_params,
                                'loss_list': loss_list},
                                step=i+1,
                                base = base,
                                dir_path = ckpt_dir)
            # re-raise so you see exactly where it happened:
            raise
        finally:
            iterator.set_description(f"Loss: {loss_list[-1]:.4f}, iter {i}")

    # final save
    if (i + 1) % checkpoint_interval != 0:        
        save_checkpoint(model, optimizer,
                                    {'param_hist': param_hist,
                                    'actual_params': actual_params,
                                    'loss_list': loss_list},
                                    step=i+1,
                                    base = base,
                                    dir_path = ckpt_dir)


    


    
    





    