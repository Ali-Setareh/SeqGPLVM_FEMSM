import hashlib
import pandas as pd
import numpy as np

def dataset_fingerprint(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    dtypes = [str(df[c].dtype) for c in cols]
    rowhash = pd.util.hash_pandas_object(df, index=True, categorize=True)

    h = hashlib.sha256()
    h.update(np.int64(len(df)).tobytes())
    h.update(",".join(cols).encode())
    h.update(",".join(dtypes).encode())
    h.update(rowhash.values.tobytes())
    return {
        "algo": "sha256",
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "hexdigest": h.hexdigest(),
    }