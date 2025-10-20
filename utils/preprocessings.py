import json
import torch
import pandas as pd
from pathlib import Path

def get_training_tensors(df: pd.DataFrame, 
                         id_col: str = "patient_id", 
                         K: int | None = None,
                         time_col: str  = "t", 
                         treatment_col: str = "D", 
                         covariate_cols_prefix: str = "x", 
                         treatment_lag: int = 1) -> tuple[torch.FloatTensor, torch.FloatTensor, dict[int, int]]:
    """
    Convert a DataFrame into training tensors for a SeqGPLVM model.

    Parameters:
    - df: pandas DataFrame containing the data.
    - id_col: str, name of the column representing individual IDs.
    - time_col: str, name of the column representing time steps.
    - treatment_col: str, name of the column representing treatment assignments.
    - covariate_cols_prefix: str, the prefix of the covariate columns e.g. x0, x1, x2, etc --> prefix = x.
    - treatment_lag: int, number of time steps to lag the treatment variable.
    
    Returns:
    - X_tensor: torch.FloatTensor of shape (N, T, K + treatment_lag) with covariates.
    - A_tensor: torch.FloatTensor of shape (N, T) with treatments.
    
    """
   

    # Ensure DataFrame is sorted by id and time
    df = df.sort_values([id_col, time_col]).copy()

    # contiguous 0..N-1 row indices for patients
    uniq_ids = df[id_col].unique()
    id2row = {pid: i for i, pid in enumerate(uniq_ids)}
    N = len(uniq_ids)
    T_max = int(df[time_col].max())

    # covariate columns
    x_cols = [c for c in df.columns if c.startswith(covariate_cols_prefix)]
    if K is None:
        K = len(x_cols)
    else:
        x_cols = x_cols[:K]

    # allocate
    A = torch.full((N, T_max), float("nan"))
    last_dim = K + treatment_lag
    X = torch.full((N, T_max, last_dim), float("nan"))

    for pid, g in df.groupby(id_col):
        r = id2row[pid]
        t_idx = g[time_col].to_numpy() - 1
        A[r, t_idx] = torch.as_tensor(g[treatment_col].to_numpy(), dtype=torch.float32)

        x_vals = torch.as_tensor(g[x_cols].to_numpy(), dtype=torch.float32)
        X[r, t_idx, :K] = x_vals

        prev_treat_cols = [f"lag{treatment_lag}_{treatment_col}" for treatment_lag in range(1, treatment_lag+1)]
        X[r, t_idx, K:] = torch.as_tensor(g[prev_treat_cols].to_numpy(), dtype=torch.float32)

    return X.contiguous(), A.contiguous(), id2row 


def grid_helper(a, b):
    nrow_a = a.size()[0]
    nrow_b = b.size()[0]
    ncol_b = b.size()[1]
    x = a.repeat(nrow_b, 1)
    y = b.repeat(1, nrow_a).view(-1, ncol_b)
    return x, y

class FeatureStandardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X):  # X: [N, T, C] torch.FloatTensor
        # Flatten N and T; drop rows with any NaNs (from lags, etc.)
        X2 = X.view(-1, X.size(-1))
        mask = ~torch.isnan(X2).any(dim=1)
        X2 = X2[mask]
        mean = X2.mean(dim=0)
        std = X2.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(mean, std)

    def transform(self, X):
        return (X - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.detach().cpu().tolist(),
                "std": self.std.detach().cpu().tolist()}

    @classmethod
    def from_dict(cls, d, device=None, dtype=None):
        m = torch.tensor(d["mean"], device=device, dtype=dtype)
        s = torch.tensor(d["std"],  device=device, dtype=dtype)
        return cls(m, s)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)