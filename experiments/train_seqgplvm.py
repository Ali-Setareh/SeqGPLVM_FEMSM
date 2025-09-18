import torch 
import json 
import os
import numpy as np
from utils.checkpoints import save_checkpoint, get
latent_dim = 1
num_inducing = 50
num_inducing_hidden = 5 #7

def train_seqgplvm():
    from models.seqgplvm import SeqGPLVM
    from utils.checkpoints import Checkpoints
    from utils.data import Data
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = Data()
    X, Y = data.load_data()

    model = SeqGPLVM(X, Y, latent_dim, num_inducing, num_inducing_hidden, device=device).to(device)

    checkpoints = Checkpoints(model, "seqgplvm", save_dir="checkpoints")

    model.fit(num_iters=1000, lr=0.01, checkpoints=checkpoints, checkpoint_interval=100)