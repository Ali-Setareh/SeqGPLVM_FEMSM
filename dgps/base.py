from typing import Dict, Any, Iterable, List
import numpy as np
import pandas as pd
import re

def rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))

def make_stem(dgp: str, params: dict) -> str:
    """
    Create a filename stem from DGP name and params dict.
    Includes all params as key-value pairs.
    Floats are rounded, lists/arrays are flattened.
    """
    parts = [dgp]

    for key, val in sorted(params.items()):  # sort → consistent order
        if isinstance(val, float):
            val = round(val, 3)  # round floats to 3 decimals
        elif isinstance(val, (list, tuple)):
            # join lists into compact form, e.g. beta[-0.5_-0.5_1.0]
            val = "_".join(str(round(v, 3)) if isinstance(v, float) else str(v) for v in val)
        elif val is None:
            val = "none"

        # sanitize for filenames (remove spaces, brackets, commas, etc.)
        sval = re.sub(r"[^A-Za-z0-9._-]", "", str(val))
        parts.append(f"{key}{sval}")

    return "_".join(parts)

def add_lag_columns(df: pd.DataFrame, cols: Iterable[str], group_col: str, time_col: str, max_lag: int, treatment_col: bool = False) -> pd.DataFrame:
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
            if treatment_col:
                df[f"lag{L}_{col}"] = df[f"lag{L}_{col}"].fillna(0)  # fill NaN with 0 for treatment lags

    return df

def make_equicorr_cov(p: int, offdiag: float) -> np.ndarray:
    """covariance with 1 on the diagonal and `offdiag` off-diagonal."""
    if p <= 0:
        raise ValueError("p must be positive.")
    Sigma = np.full((p, p), offdiag, dtype=float)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma
