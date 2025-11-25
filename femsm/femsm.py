from rpy2.robjects import r
from rpy2.robjects.packages import importr, isinstalled


def _ensure_r_packages(pkgs):
    utils = importr("utils")
    to_install = [p for p in pkgs if not isinstalled(p)]
    if to_install:
        utils.install_packages(r.c(*to_install), repos="https://cloud.r-project.org")

# Make sure needed R packages exist
_ensure_r_packages(["fs", "arrow", "jsonlite", "dplyr", "stringr", "fixest", "sandwich"])

# Define the R function performing the MSM estimation using fix effects
r('''
msm_from_py <- function(df,
                        train_ids,
                        k_last = 4,
                        a_val = NA_real_,
                        data_id = NA_character_,
                        x_cols = NULL) {
  suppressPackageStartupMessages({
    library(dplyr)
    library(fixest)
    library(sandwich)
  })

  clip01 <- function(p, eps = 1e-6) pmax(eps, pmin(1 - eps, p))
  train_ids <- as.character(train_ids)

  if (!("patient_id" %in% names(df))) stop("df must contain 'patient_id'.")
  if (!("t" %in% names(df))) stop("df must contain 't'.")
  if (!("D" %in% names(df))) stop("df must contain 'D'.")
  if (!("Y" %in% names(df))) stop("df must contain 'Y'.")

  df$patient_id <- as.character(df$patient_id)

  if (is.null(x_cols)) {
    x_cols <- grep("^x", names(df), value = TRUE)
  } else {
    x_cols <- intersect(as.character(x_cols), names(df))
  }

  # Keep all train units; FE fit will subset internally
  df_fe <- df %>%
    dplyr::filter(patient_id %in% train_ids) %>%
    dplyr::arrange(patient_id, t)

  # Variable-treatment vs monotone units
  tmp <- df_fe %>%
    dplyr::group_by(patient_id) %>%
    dplyr::summarise(meanD = mean(D), .groups="drop")
  var_pids    <- tmp$patient_id[!(tmp$meanD %in% c(0, 1))]
  always0_pids <- tmp$patient_id[tmp$meanD == 0]  
  always1_pids <- tmp$patient_id[tmp$meanD == 1]   

  T_final <- max(df_fe$t)
  if (k_last < 1) stop("k_last must be >= 1")
  mask_lastk <- df_fe$t >= (T_final - (k_last - 1)) & df_fe$t <= T_final

  need_lags <- !all(c("lag1_D","lag2_D","lag3_D") %in% names(df_fe))
  if (need_lags) {
    df_fe <- df_fe %>%
      dplyr::arrange(patient_id, t) %>%
      dplyr::group_by(patient_id) %>%
      dplyr::mutate(
        lag1_D = dplyr::lag(D, 1, default = 0),
        lag2_D = dplyr::lag(D, 2, default = 0),
        lag3_D = dplyr::lag(D, 3, default = 0)
      ) %>%
      dplyr::ungroup()
  }
  df_fe$lag_sum3 <- df_fe$lag1_D + df_fe$lag2_D + df_fe$lag3_D

  df_msm <- dplyr::filter(df_fe, t == T_final)

  # Numerator model
  num_mod <- stats::glm(D ~ lag1_D, data = df_fe, family = stats::binomial())
  p_num   <- clip01(stats::predict(num_mod, type = "response"))

  # Denominator no-FE
  f_no <- stats::reformulate(termlabels = c("lag1_D", x_cols), response = "D")
  den_no <- stats::glm(f_no, data = df_fe, family = stats::binomial())
  p_no   <- clip01(stats::predict(den_no, type = "response"))

  # Denominator FE (only variable-treatment units)
  df_fe_FE <- df_fe[df_fe$patient_id %in% var_pids, , drop = FALSE]
  rhs      <- paste(c("lag1_D", x_cols), collapse = " + ")
  den_fe   <- fixest::feglm(stats::as.formula(paste0("D ~ ", rhs, " | patient_id")),
                            data = df_fe_FE, family = stats::binomial())
  p_fe     <- clip01(stats::predict(den_fe, type = "response"))

  idx_fe <- which(df_fe$patient_id %in% var_pids)
  p_fe_full <- rep(NA_real_, nrow(df_fe))
  p_fe_full[idx_fe] <- p_fe

  p_fe_impute <- p_fe_full
  if (length(always0_pids) > 0) {
    p_fe_impute[df_fe$patient_id %in% always0_pids] <- 0.01
  }
  if (length(always1_pids) > 0) {
    p_fe_impute[df_fe$patient_id %in% always1_pids] <- 0.99
  }
  p_fe_impute <- clip01(p_fe_impute)  # ensure in (0,1)
  # --------------------------------------------------------#

  p_true <- if ("p_true" %in% names(df_fe)) clip01(df_fe$p_true) else rep(NA_real_, nrow(df_fe))

  per_period_w <- function(pn, pd, D) ifelse(D == 1, pn/pd, (1 - pn)/(1 - pd))

  w_true_it <- if (all(!is.na(p_true))) per_period_w(p_num, p_true, df_fe$D) else rep(NA_real_, nrow(df_fe))
  w_no_it   <- per_period_w(p_num, p_no,   df_fe$D)

  # FE, drop monotone 
  p_num_FE  <- p_num[df_fe$patient_id %in% var_pids]
  w_fe_it   <- per_period_w(p_num_FE, p_fe, df_fe_FE$D)

  # FE-impute, keep all units with imputed ps for monotone
  w_fe_imp_it <- per_period_w(p_num, p_fe_impute, df_fe$D)

  lastk_prod <- function(ids, w, mask) {
    d <- data.frame(patient_id = ids[mask], w = w[mask])
    stats::aggregate(w ~ patient_id, data = d, FUN = prod)
  }

  W_true_tbl <- if (all(!is.na(w_true_it))) lastk_prod(df_fe$patient_id, w_true_it, mask_lastk) else
                  data.frame(patient_id = df_msm$patient_id, w = NA_real_)
  W_no_tbl   <- lastk_prod(df_fe$patient_id,   w_no_it,   mask_lastk)

  mask_lastk_FE <- df_fe_FE$t >= (T_final - (k_last - 1)) & df_fe_FE$t <= T_final
  W_fe_tbl      <- lastk_prod(df_fe_FE$patient_id, w_fe_it, mask_lastk_FE)

  # last-k products for imputed FE weights on all units
  W_fe_imp_tbl  <- lastk_prod(df_fe$patient_id, w_fe_imp_it, mask_lastk)

  df_msm$patient_id     <- as.character(df_msm$patient_id)
  W_true_tbl$patient_id <- as.character(W_true_tbl$patient_id)
  W_no_tbl$patient_id   <- as.character(W_no_tbl$patient_id)
  W_fe_tbl$patient_id   <- as.character(W_fe_tbl$patient_id)
  W_fe_imp_tbl$patient_id <- as.character(W_fe_imp_tbl$patient_id)   # NEW

  w_true <- if (all(!is.na(w_true_it))) W_true_tbl$w[ match(df_msm$patient_id, W_true_tbl$patient_id) ]
            else rep(NA_real_, nrow(df_msm))
  w_no   <- W_no_tbl$w  [ match(df_msm$patient_id, W_no_tbl$patient_id) ]

  df_msm_FE <- df_msm[df_msm$patient_id %in% var_pids, , drop=FALSE]
  w_fe      <- W_fe_tbl$w[ match(as.character(df_msm_FE$patient_id), W_fe_tbl$patient_id) ]

  # imputed FE weights align with all df_msm rows
  w_fe_imp  <- W_fe_imp_tbl$w[ match(df_msm$patient_id, W_fe_imp_tbl$patient_id) ]

  fit_wls <- function(data, w) stats::lm(Y ~ D + lag_sum3, data = data, weights = w)

  res_true <- if (all(!is.na(w_true))) fit_wls(df_msm, w_true) else NULL
  res_no   <- fit_wls(df_msm, w_no)
  res_fe   <- if (nrow(df_msm_FE) > 0) fit_wls(df_msm_FE, w_fe) else NULL
  res_fe_impute <- fit_wls(df_msm, w_fe_imp)   # NEW

  # HC2 SEs for 90% CIs
  tau_f_true_se <- NA_real_
  tau_c_true_se <- NA_real_
  tau_f_no_se   <- NA_real_
  tau_c_no_se   <- NA_real_
  tau_f_fe_se   <- NA_real_
  tau_c_fe_se   <- NA_real_
  tau_f_fe_impute_se <- NA_real_   # NEW
  tau_c_fe_impute_se <- NA_real_   # NEW

  if (!is.null(res_true)) {
    vc_true <- sandwich::vcovHC(res_true, type = "HC2")
    se_true <- sqrt(diag(vc_true))
    tau_f_true_se <- se_true[["D"]]
    tau_c_true_se <- se_true[["lag_sum3"]]
  }

  vc_no <- sandwich::vcovHC(res_no, type = "HC2")
  se_no <- sqrt(diag(vc_no))
  tau_f_no_se <- se_no[["D"]]
  tau_c_no_se <- se_no[["lag_sum3"]]

  if (!is.null(res_fe)) {
    vc_fe <- sandwich::vcovHC(res_fe, type = "HC2")
    se_fe <- sqrt(diag(vc_fe))
    tau_f_fe_se <- se_fe[["D"]]
    tau_c_fe_se <- se_fe[["lag_sum3"]]
  }

  # SEs for FE-impute
  if (!is.null(res_fe_impute)) {
    vc_fe_imp <- sandwich::vcovHC(res_fe_impute, type = "HC2")
    se_fe_imp <- sqrt(diag(vc_fe_imp))
    tau_f_fe_impute_se <- se_fe_imp[["D"]]
    tau_c_fe_impute_se <- se_fe_imp[["lag_sum3"]]
  }

  tau_f_true <- if (!is.null(res_true)) stats::coef(res_true)[["D"]] else NA_real_
  tau_c_true <- if (!is.null(res_true)) stats::coef(res_true)[["lag_sum3"]] else NA_real_
  tau_f_no   <- stats::coef(res_no)[["D"]]
  tau_c_no   <- stats::coef(res_no)[["lag_sum3"]]
  tau_f_fe   <- if (!is.null(res_fe)) stats::coef(res_fe)[["D"]] else NA_real_
  tau_c_fe   <- if (!is.null(res_fe)) stats::coef(res_fe)[["lag_sum3"]] else NA_real_

  # point estimates for FE-impute
  tau_f_fe_impute <- if (!is.null(res_fe_impute)) stats::coef(res_fe_impute)[["D"]] else NA_real_
  tau_c_fe_impute <- if (!is.null(res_fe_impute)) stats::coef(res_fe_impute)[["lag_sum3"]] else NA_real_

  N <- length(train_ids)
  T_val <- T_final
  rho <- as.integer(N / T_val)
  p_count <- length(x_cols)

  dplyr::tibble(
    data_id = data_id,
    N = N, T = T_val, rho = rho,
    a = a_val, p = p_count,
    tau_f_true = tau_f_true, tau_c_true = tau_c_true,
    tau_f_fe   = tau_f_fe,   tau_c_fe   = tau_c_fe,             # FE, drop monotone
    tau_f_fe_impute = tau_f_fe_impute, tau_c_fe_impute = tau_c_fe_impute, 
    tau_f_no_fe = tau_f_no,  tau_c_no_fe = tau_c_no,
    tau_f_true_se = tau_f_true_se, tau_c_true_se = tau_c_true_se,
    tau_f_fe_se   = tau_f_fe_se,   tau_c_fe_se   = tau_c_fe_se,
    tau_f_fe_impute_se = tau_f_fe_impute_se, tau_c_fe_impute_se = tau_c_fe_impute_se, 
    tau_f_no_fe_se = tau_f_no_se,  tau_c_no_fe_se = tau_c_no_se
  )
}
''')
