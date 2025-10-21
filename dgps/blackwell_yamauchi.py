from typing import Dict, Any
import numpy as np
import pandas as pd


from .base import add_lag_columns, make_equicorr_cov, rng_from_seed

def simulate(params: Dict[str, Any]) -> pd.DataFrame:
    """
    Blackwell & Yamauchi (2024) simulation DGP.

    Inputs (params dict):
    - n: int                   # number of units
    - T: int                   # number of time points
    - p: int                   # number of covariates (2 or 4 in the paper; any p>=1 works)
    - a: float                 # Uniform[-a, a] for alpha_i; paper uses a ∈ {1, 2}
    - phi: float               # coefficient on lagged treatment in treatment model (paper: 0.3)
    - beta: list[float]        # length p; example (−0.5, −0.5) or (−0.5, −0.5, 1.0, −0.5)
    - tau_F: float             # effect of final treatment on Y (paper: 1.0)
    - tau_C: float             # effect per lagged treatment in last 3 periods (paper: 0.3)
    - gamma: list[float]       # length p; example (1.0, 0.5) or (1.0, 0.5, 1.0, 1.0)
    - mean_x: float            # mean for each covariate dimension (paper uses −0.5 per dim)
    - offdiag: float           # off-diagonal Σ entries for X (paper: 0.2)
    - sigma_eps: float         # sd of noise in Y (paper uses 1.0)
    - max_lag_x: int           # how many covariate lags to materialize as columns (naming scheme)
    - seed: int | None         # RNG seed

    Returns:
    Long-format DataFrame with columns:
    ['patient_id','t','D','Y','x0',...,'x{p-1}', plus lag columns 'lag1_x0',..., 'lagL_x{p-1}' if requested]

    Notes aligning with the paper:
    - Treatment model: D_it ~ Bernoulli(expit(alpha_i + phi*D_{i,t-1} + beta^T X_it))
    - X_it ~ MVN(mean=-0.5*1_p, Σ with 1 on diag and 0.2 off diag)
    - Outcome (single endpoint): Y_i = alpha_i + tau_F*D_{iT} + tau_C*sum_{t=T-3}^{T-1} D_it + gamma^T \bar{X}_i + eps
      (we attach Y_i only at time T row, NaN otherwise)
    """
    # --- Pull params with sensible defaults from the paper ---
    n         = int(params.get("n", 1000))
    T         = int(params.get("T", n // 10 if n >= 10 else 10))
    p         = int(params.get("p", 2))
    a         = float(params.get("a", 1.0))
    phi       = float(params.get("phi", 0.3))
    beta      = np.asarray(params.get("beta", [-0.5, -0.5] if p == 2 else [-0.5, -0.5, 1.0, -0.5]), dtype=float)
    tau_F     = float(params.get("tau_F", 1.0))
    tau_C     = float(params.get("tau_C", 0.3))
    gamma     = np.asarray(params.get("gamma", [1.0, 0.5] if p == 2 else [1.0, 0.5, 1.0, 1.0]), dtype=float)
    mean_x    = float(params.get("mean_x", -0.5))
    offdiag   = float(params.get("offdiag", 0.2))
    sigma_eps = float(params.get("sigma_eps", 1.0))
    max_lag_x = int(params.get("max_lag_x", 0))
    max_lag_d = int(params.get("max_lag_d", 0))
    seed      = params.get("seed", None)

    if beta.shape[0] != p or gamma.shape[0] != p:
        raise ValueError(f"beta and gamma must have length p={p}.")

    if T < 2:
        raise ValueError("T must be at least 2.")

    rng = rng_from_seed(seed) #rng or np.random.default_rng(seed) 

    # --- Draw unit-level heterogeneity alpha_i ~ Uniform[-a, a] ---
    alpha = rng.uniform(-a, a, size=n)  # (n,)

    # --- Build covariance and draw covariates X_it ~ MVN(mean_x * 1_p, Σ) ---
    mu = np.full(p, mean_x, dtype=float)
    Sigma = make_equicorr_cov(p, offdiag=offdiag)

    # Shape X: (n, T, p)
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=(n, T))

    # --- Generate treatments with 1-lag dependence ---
    D = np.zeros((n, T), dtype=int)
    D_lag = np.zeros(n, dtype=float)  # D_{i,0} = 0
    eta_true = np.empty((n, T), dtype=float)     
    p_true   = np.empty((n, T), dtype=float)      

    def expit(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    for t in range(T):
        lin = alpha + phi * D_lag + X[:, t, :].dot(beta)
        eta_true[:, t] = lin
        p_t = expit(lin)
        p_true[:, t] = p_t
        D[:, t] = rng.binomial(1, p_t)
        D_lag = D[:, t]

    # --- Outcome: one endpoint Y_i (attach to time T row) ---
    # Sum of last three pre-final treatments (handle small T gracefully)
    k = min(3, max(T - 1, 0))
    if k == 0:
        lag_sum = np.zeros(n)
    else:
        # collect the last k periods before final (exclude final period)
        lag_indices = list(range(T - 1 - k, T - 1))
        lag_sum = D[:, lag_indices].sum(axis=1)

    Xbar = X.mean(axis=1)                # (n, p)
    eps  = rng.normal(0.0, sigma_eps, size=n)
    Y    = alpha + tau_F * D[:, -1] + tau_C * lag_sum + Xbar.dot(gamma) + eps  # (n,)

    # --- Assemble long DataFrame ---
    rows = []
    for i in range(n):
        for t in range(T):
            rec = {
                "patient_id": i,
                "t": t + 1,
                "D": int(D[i, t]),
                "Y": float(Y[i]) if (t == T - 1) else np.nan,
                "alpha": float(alpha[i]),  # for inspection purposes
                "eta_true": float(eta_true[i, t]),     
                "p_true": float(p_true[i, t]),
            }
            for j in range(p):
                rec[f"x{j}"] = float(X[i, t, j])
            rows.append(rec)

    df = pd.DataFrame(rows)

    # Materialize requested covariate/treatment lags 
    x_cols = [f"x{j}" for j in range(p)]
    if max_lag_x > 0:
        df = add_lag_columns(df, cols=x_cols, group_col="patient_id", time_col="t", max_lag=max_lag_x)
    if max_lag_d > 0:
        df = add_lag_columns(df, cols=["D"], group_col="patient_id", time_col="t", max_lag=max_lag_d, treatment_col=True)

    return df