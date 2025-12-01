import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List

def seqgplvm_msm_from_py_py(
    df: pd.DataFrame,
    train_ids,
    propensity_scores_cols: List[str],
    k_last: int = 4,
    a_val: float | None = None,
    data_id: str | None = None,
    x_cols=None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Python reimplementation of the R function seqgplvm_msm_from_py().

    Returns a one-row DataFrame with the same fields as the R tibble:
      tau_f_fe, tau_c_fe, tau_f_fe_impute, tau_c_fe_impute,
      and their HC2 standard errors, plus metadata.
    """
    
    def per_period_w(pn, pd, D_vec):
        return np.where(D_vec == 1, pn / pd, (1.0 - pn) / (1.0 - pd))
    
    # Copy & normalise types
    df = df.copy()
    
    # --- subset to train_ids & sort (R: filter + arrange(patient_id, t)) ------
    df_msm = df[df["patient_id"].isin(train_ids)].copy()
    df_msm = df_msm.sort_values(["patient_id", "t"]).reset_index(drop=True)
    N = df_msm["patient_id"].nunique()

    # --- classify monotone vs variable-treatment units ------------------------
    meanD = df_msm.groupby("patient_id")["D"].mean()
    always0_pids = meanD.index[meanD == 0]
    always1_pids = meanD.index[meanD == 1]
    var_pids = meanD.index[(meanD != 0) & (meanD != 1)]
    

    df_msm["lag_sum3"] = df_msm["lag1_D"] + df_msm["lag2_D"] + df_msm["lag3_D"]

    
    # --- numerator model: glm(D ~ lag1_D, family=binomial) -------------------
    X_num = sm.add_constant(df_msm[["lag1_D"]].astype(float))
    y_num = df_msm["D"].astype(float)
    num_mod = sm.GLM(y_num, X_num, family=sm.families.Binomial())
    res_num = num_mod.fit()

    p_num = res_num.predict(X_num)
    p_num = np.clip(p_num, eps, 1.0 - eps)
    df_msm["p_num"] = p_num

    df_msm_always0 = df_msm[df_msm["patient_id"].isin(always0_pids)].copy()
    df_msm_always1 = df_msm[df_msm["patient_id"].isin(always1_pids)].copy()

    # --- subset to variable-treatment patients only for denominator ---------------

    df_msm_var = df_msm[df_msm["patient_id"].isin(var_pids)].copy().reset_index(drop=True)
    D_all_var = df_msm_var["D"].astype(int).to_numpy()


    # denominator: SeqGPLVM propensities from given column (R: df_fe[[col]])
    w_t_var_cols = {}
    for propensity_scores_col in propensity_scores_cols:
        p_hat_var = df_msm_var[propensity_scores_col].astype(float).to_numpy()
        p_hat_var = np.clip(p_hat_var, eps, 1.0 - eps)
        df_msm_var[propensity_scores_col] = p_hat_var



        # IPTW weights for all periods
        w_t_var = per_period_w(df_msm_var["p_num"].to_numpy(), df_msm_var[propensity_scores_col].to_numpy(), D_all_var)
        w_t_var_cols[f"w_t_{propensity_scores_col}"] = w_t_var
    
    df_msm_var  = pd.concat(
        [df_msm_var, pd.DataFrame(w_t_var_cols)], axis=1
    )


    # cumulative product weights by patient
    T_final = int(df_msm["t"].max())

    mask_lastk = (df_msm["t"] >= (T_final - (k_last - 1))) & (df_msm["t"] <= T_final)
    df_msm_var = df_msm_var.loc[mask_lastk, :]

    w_comprod_cols = {}
    for propensity_scores_col in propensity_scores_cols:
        w_comprod_cols[f"w_cumprod_{propensity_scores_col}"] = df_msm_var.groupby("patient_id")[f"w_t_{propensity_scores_col}"].cumprod()

    df_msm_var = pd.concat(
        [df_msm_var, pd.DataFrame(w_comprod_cols)], axis=1
    )

    df_msm_var = df_msm_var[df_msm_var.t == T_final].copy().reset_index(drop=True)


    always0 = always1 = 0
    if len(always0_pids) > 0:
        always0 = 1
        df_msm_always0["p_hat"] = eps
        D_all_always0 = df_msm_always0["D"].astype(int).to_numpy()
        w_t_always0 = per_period_w(df_msm_always0["p_num"].to_numpy(), df_msm_always0["p_hat"].to_numpy(), D_all_always0)
        df_msm_always0["w_t"] = w_t_always0
        mask_lastk = (df_msm_always0["t"] >= (T_final - (k_last - 1))) & (df_msm_always0["t"] <= T_final)
        df_msm_always0 = df_msm_always0.loc[mask_lastk, :]
        df_msm_always0["w_cumprod"] = df_msm_always0.groupby("patient_id")["w_t"].cumprod()
        df_msm_always0 = df_msm_always0[df_msm_always0.t == T_final]
    
    if len(always1_pids) > 0:
        always1 = 1
        df_msm_always1["p_hat"] = 1.0 - eps
        D_all_always1 = df_msm_always1["D"].astype(int).to_numpy()
        w_t_always1 = per_period_w(df_msm_always1["p_num"].to_numpy(), df_msm_always1["p_hat"].to_numpy(), D_all_always1)
        df_msm_always1["w_t"] = w_t_always1
        mask_lastk = (df_msm_always1["t"] >= (T_final - (k_last - 1))) & (df_msm_always1["t"] <= T_final)
        df_msm_always1 = df_msm_always1.loc[mask_lastk, :]
        df_msm_always1["w_cumprod"] = df_msm_always1.groupby("patient_id")["w_t"].cumprod()
        df_msm_always1 = df_msm_always1[df_msm_always1.t == T_final]
    
    # --- weighted least squares with HC2 SEs ---------------------------------
    def fit_wls(data: pd.DataFrame, weight_col: str):
        X = sm.add_constant(data[["D", "lag_sum3"]].astype(float))
        y = data["Y"].astype(float)
        
        w = data[weight_col].astype(float).to_numpy()
        
        model = sm.WLS(y, X, weights=w)
        res = model.fit(cov_type="HC2")
        return res
    
    results = {"batch_id": []}
    items = ["tau_f_seqgplvm", "tau_c_seqgplvm", "tau_f_seqgplvm_se", "tau_c_seqgplvm_se",
             "tau_f_seqgplvm_imp", "tau_c_seqgplvm_imp", "tau_f_seqgplvm_se_imp", "tau_c_seqgplvm_se_imp"]
    for item in items:
        results[item] = []
    


    for propensity_scores_col in propensity_scores_cols:
        df_msm_always0_temp = df_msm_always1_temp = pd.DataFrame()

        res_fe_var = fit_wls(df_msm_var, f"w_cumprod_{propensity_scores_col}") 
 
        # extract coefficients & SEs (like R code)
        results["batch_id"].append(propensity_scores_col)
        results[f"tau_f_seqgplvm"].append(res_fe_var.params["D"])
        results[f"tau_c_seqgplvm"].append(res_fe_var.params["lag_sum3"])
        results[f"tau_f_seqgplvm_se"].append(res_fe_var.bse["D"])
        results[f"tau_c_seqgplvm_se"].append(res_fe_var.bse["lag_sum3"])
        
        results[f"tau_f_seqgplvm_imp"].append(res_fe_var.params["D"])
        results[f"tau_c_seqgplvm_imp"].append(res_fe_var.params["lag_sum3"])
        results[f"tau_f_seqgplvm_se_imp"].append(res_fe_var.bse["D"])
        results[f"tau_c_seqgplvm_se_imp"].append(res_fe_var.bse["lag_sum3"])
        

        # combine always0, always1, variable back for the regression with imputation
        if always0 == 1: 
            df_msm_always0_temp = df_msm_always0.rename(columns={"w_cumprod": f"w_cumprod_{propensity_scores_col}"}).copy()
        if always1 == 1:
            df_msm_always1_temp = df_msm_always1.rename(columns={"w_cumprod": f"w_cumprod_{propensity_scores_col}"}).copy()

        if always1 == 1 or always0 == 1:
            df_msm_imp = pd.concat([df_msm_var, df_msm_always0_temp, df_msm_always1_temp]).sort_values(["patient_id", "t"]).reset_index(drop=True)
            res_fe_imp = fit_wls(df_msm_imp, f"w_cumprod_{propensity_scores_col}")

            results[f"tau_f_seqgplvm_imp"][-1] = res_fe_imp.params["D"]
            results[f"tau_c_seqgplvm_imp"][-1] = res_fe_imp.params["lag_sum3"]
            results[f"tau_f_seqgplvm_se_imp"][-1] = res_fe_imp.bse["D"]
            results[f"tau_c_seqgplvm_se_imp"][-1] = res_fe_imp.bse["lag_sum3"]
    
    
    # --- metadata ------------------------------------------------------------
    T_val = T_final
    rho = int(N / T_val) if T_val > 0 else np.nan
    p_count = len(x_cols)
    out = pd.DataFrame(
        
            {
                **results,
                "data_id": data_id,
                "a": a_val,
                "N": N,
                "T": T_val,
                "rho": rho,
                "p": p_count,
                "always0": len(always0_pids),
                "always1": len(always1_pids)
            }
        
    )
    return out

