import pandas as pd
from rpy2.robjects import r, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2 import rinterface
import femsm 

def msm_estimates_from_df(df: pd.DataFrame,
                          train_ids,
                          *,
                          k_last: int = 4,
                          a_val: float | None = None,
                          data_id: str | None = None,
                          x_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Runs the R msm_from_py() on a single dataset already in Python.

    Returns a 1-row pandas DataFrame with the tau_* and metadata.
    """
    # Convert inputs to R types inside a local converter context
    with localconverter(default_converter + pandas2ri.converter):
        r_df = pandas2ri.py2rpy(df)
        r_train = pandas2ri.py2rpy(pd.Series(list(train_ids)))
        r_xcols = pandas2ri.py2rpy(pd.Series(x_cols)) if x_cols is not None else rinterface.NULL

    # Prepare defaulted scalars
    k_last = int(k_last)
    a_val = float(a_val) if a_val is not None else float("nan")
    data_id = data_id if data_id is not None else ""

    # Call the R function
    res_r = r["msm_from_py"](r_df, r_train, k_last, a_val, data_id, r_xcols)

    # Convert back to pandas
    with localconverter(default_converter + pandas2ri.converter):
        res_py = pandas2ri.rpy2py(res_r)

    return res_py
