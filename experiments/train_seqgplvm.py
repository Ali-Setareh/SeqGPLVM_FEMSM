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
                   treatment_lag=1,
                   pid_col = "patient_id",
                   time_col = "t",
                   metadata_cov_num_key = "p",
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                   ):
    
    df_with_lags = df[df[time_col]!=0]

    N = df_with_lags[pid_col].nunique()
    K = df_meta_data[metadata_cov_num_key]
    max_T = df_with_lags[time_col].max()

    




    