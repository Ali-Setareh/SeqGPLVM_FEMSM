import torch 
import json 
import os
import numpy as np
from models.SeqGPLVM import SeqGPLVM
from utils.checkpoints import save_checkpoint 
from utils.inspectors import get_actuals_via_getters 


def train_seqgplvm(df,
                   df_meta_data,
                   latent_dim=1,
                   num_inducing=50, 
                   num_inducing_hidden=5,
                   pid_col = "patient_id",
                   time_col = "t",
                   covariate_num_in_metadata_key = "p",
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                   ):
    
    df_with_lags = df[df[time_col]!=0]

    N = df_with_lags[pid_col].nunique()
    K = df_meta_data[covariate_num_in_metadata_key]

    model = SeqGPLVM(X, Y, latent_dim, num_inducing, num_inducing_hidden, device=device).to(device)

    checkpoints = Checkpoints(model, "seqgplvm", save_dir="checkpoints")

    model.fit(num_iters=1000, lr=0.01, checkpoints=checkpoints, checkpoint_interval=100)