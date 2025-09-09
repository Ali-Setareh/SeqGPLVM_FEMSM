from typing import Dict, Any, Iterable, List
import numpy as np
import pandas as pd

def rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))

def add_lag_columns(df: pd.DataFrame, cols: Iterable[str], group_col: str, time_col: str, max_lag: int) -> pd.DataFrame:
    """
    Add lag columns for `cols` within groups defined by `group_col`, ordered by `time_col`.
    New columns follow the scheme: lag{L}_{col}, e.g., lag1_x0, lag2_x1, ...
    """
    if max_lag <= 0:
        return df

    df = df.sort_values([group_col, time_col]).copy()
    grp = df.groupby(group_col, sort=False)

    for col in cols:
        for L in range(1, max_lag + 1):
            df[f"lag{L}_{col}"] = grp[col].shift(L)

    return df

def make_equicorr_cov(p: int, offdiag: float) -> np.ndarray:
    """covariance with 1 on the diagonal and `offdiag` off-diagonal."""
    if p <= 0:
        raise ValueError("p must be positive.")
    Sigma = np.full((p, p), offdiag, dtype=float)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma
