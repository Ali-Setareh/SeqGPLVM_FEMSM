from rpy2.robjects import r
from rpy2.robjects.packages import importr, isinstalled


def _ensure_r_packages(pkgs):
    utils = importr("utils")
    to_install = [p for p in pkgs if not isinstalled(p)]
    if to_install:
        utils.install_packages(r.c(*to_install), repos="https://cloud.r-project.org")

# Make sure needed R packages exist
_ensure_r_packages(["fs", "arrow", "jsonlite", "dplyr", "stringr", "sandwich"])


# Define the R function performing the MSM estimation using SeqGPLVM propensities
r('''
seqgplvm_msm_from_py <- function(df,
                                 train_ids,
                                 propensity_scores_col,
                                 k_last = 4,
                                 a_val = NA_real_,
                                 data_id = NA_character_,
                                 x_cols = NULL) {
  suppressPackageStartupMessages({
    library(dplyr)
    library(sandwich)
  })

  clip01 <- function(p, eps = 1e-6) pmax(eps, pmin(1 - eps, p))
  train_ids <- as.character(train_ids)

  # basic checks
  req <- c("patient_id", "t", "D", "Y", propensity_scores_col)
  miss <- setdiff(req, names(df))
  if (length(miss) > 0L) {
    stop("df is missing required columns: ", paste(miss, collapse = ", "))
  }

  df$patient_id <- as.character(df$patient_id)

  # covariate columns (only used in metadata for now)
  if (is.null(x_cols)) {
    x_cols <- grep("^x", names(df), value = TRUE)
  } else {
    x_cols <- intersect(as.character(x_cols), names(df))
  }

  # subset to train_ids
  df_fe <- df %>%
    dplyr::filter(patient_id %in% train_ids) %>%
    dplyr::arrange(patient_id, t)

  if (nrow(df_fe) == 0L) {
    stop("No rows left after filtering to train_ids in seqgplvm_msm_from_py.")
  }

  # classify monotone vs variable-treatment units
  tmp <- df_fe %>%
    dplyr::group_by(patient_id) %>%
    dplyr::summarise(meanD = mean(D), .groups = "drop")
  var_pids     <- tmp$patient_id[!(tmp$meanD %in% c(0, 1))]
  always0_pids <- tmp$patient_id[tmp$meanD == 0]
  always1_pids <- tmp$patient_id[tmp$meanD == 1]

  T_final <- max(df_fe$t)
  if (k_last < 1) stop("k_last must be >= 1")
  mask_lastk <- df_fe$t >= (T_final - (k_last - 1)) & df_fe$t <= T_final

  # build lags if needed
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

  # final-time rows
  df_msm <- dplyr::filter(df_fe, t == T_final)

  # metadata (grab from df if available)
  training_id <- if ("training_id" %in% names(df)) unique(df$training_id)[1] else NA_character_
  seed        <- if ("seed"        %in% names(df)) unique(df$seed)[1]        else NA_integer_

  # numerator model: stabilized weights
  num_mod <- stats::glm(D ~ lag1_D, data = df_fe, family = stats::binomial())
  p_num   <- clip01(stats::predict(num_mod, type = "response"))

  # denominator: SeqGPLVM propensities from the given column
  p_hat <- clip01(df_fe[[propensity_scores_col]])

  per_period_w <- function(pn, pd, D) ifelse(D == 1, pn/pd, (1 - pn)/(1 - pd))

  # --- (1) DROP monotone units: only use var_pids -----------------------------
  df_fe_FE <- df_fe[df_fe$patient_id %in% var_pids, , drop = FALSE]
  if (nrow(df_fe_FE) > 0L) {
    idx_FE    <- which(df_fe$patient_id %in% var_pids)
    p_num_FE  <- p_num[idx_FE]
    p_hat_FE  <- p_hat[idx_FE]
    w_fe_it   <- per_period_w(p_num_FE, p_hat_FE, df_fe_FE$D)
  } else {
    w_fe_it <- numeric(0)
  }

  # --- (2) IMPUTE propensities for monotone units -----------------------------
  p_hat_imp <- p_hat
  if (length(always0_pids) > 0L) {
    p_hat_imp[df_fe$patient_id %in% always0_pids] <- 0.01
  }
  if (length(always1_pids) > 0L) {
    p_hat_imp[df_fe$patient_id %in% always1_pids] <- 0.99
  }
  p_hat_imp <- clip01(p_hat_imp)

  w_fe_imp_it <- per_period_w(p_num, p_hat_imp, df_fe$D)

  # helper: product over last k periods
  lastk_prod <- function(ids, w, mask) {
    d <- data.frame(patient_id = ids[mask], w = w[mask])
    stats::aggregate(w ~ patient_id, data = d, FUN = prod)
  }

  # 1) DROP monotone: weights only for var_pids
  if (length(w_fe_it) > 0L) {
    mask_lastk_FE <- df_fe_FE$t >= (T_final - (k_last - 1)) & df_fe_FE$t <= T_final
    W_fe_tbl      <- lastk_prod(df_fe_FE$patient_id, w_fe_it, mask_lastk_FE)
    W_fe_tbl$patient_id <- as.character(W_fe_tbl$patient_id)
  } else {
    W_fe_tbl <- data.frame(patient_id = df_msm$patient_id[FALSE], w = numeric(0))
  }

  # 2) IMPUTED: weights for all train_ids
  W_fe_imp_tbl <- lastk_prod(df_fe$patient_id, w_fe_imp_it, mask_lastk)
  W_fe_imp_tbl$patient_id <- as.character(W_fe_imp_tbl$patient_id)

  # align with final-time rows
  df_msm$patient_id <- as.character(df_msm$patient_id)

  df_msm_FE <- df_msm[df_msm$patient_id %in% var_pids, , drop = FALSE]
  w_fe <- if (nrow(df_msm_FE) > 0L) {
    W_fe_tbl$w[ match(df_msm_FE$patient_id, W_fe_tbl$patient_id) ]
  } else {
    numeric(0)
  }

  w_fe_imp <- W_fe_imp_tbl$w[ match(df_msm$patient_id, W_fe_imp_tbl$patient_id) ]

  fit_wls <- function(data, w) stats::lm(Y ~ D + lag_sum3, data = data, weights = w)

  res_fe     <- if (nrow(df_msm_FE) > 0L) fit_wls(df_msm_FE, w_fe) else NULL
  res_fe_imp <- fit_wls(df_msm, w_fe_imp)

  # HC2 SEs
  tau_f_fe_se <- tau_c_fe_se <- NA_real_
  tau_f_fe_impute_se <- tau_c_fe_impute_se <- NA_real_

  if (!is.null(res_fe)) {
    vc_fe <- sandwich::vcovHC(res_fe, type = "HC2")
    se_fe <- sqrt(diag(vc_fe))
    tau_f_fe_se <- se_fe[["D"]]
    tau_c_fe_se <- se_fe[["lag_sum3"]]
  }

  if (!is.null(res_fe_imp)) {
    vc_fe_imp <- sandwich::vcovHC(res_fe_imp, type = "HC2")
    se_fe_imp <- sqrt(diag(vc_fe_imp))
    tau_f_fe_impute_se <- se_fe_imp[["D"]]
    tau_c_fe_impute_se <- se_fe_imp[["lag_sum3"]]
  }

  tau_f_fe        <- if (!is.null(res_fe))     stats::coef(res_fe)[["D"]]        else NA_real_
  tau_c_fe        <- if (!is.null(res_fe))     stats::coef(res_fe)[["lag_sum3"]] else NA_real_
  tau_f_fe_impute <- if (!is.null(res_fe_imp)) stats::coef(res_fe_imp)[["D"]]    else NA_real_
  tau_c_fe_impute <- if (!is.null(res_fe_imp)) stats::coef(res_fe_imp)[["lag_sum3"]] else NA_real_

  # meta
  N <- length(unique(df_fe$patient_id))
  T_val <- T_final
  rho <- as.integer(N / T_val)
  p_count <- length(x_cols)

  dplyr::tibble(
    data_id = data_id,
    training_id = training_id,
    seed = seed,
    batch = propensity_scores_col,
    N = N, T = T_val, rho = rho,
    a = a_val, p = p_count,
    tau_f_fe = tau_f_fe, tau_c_fe = tau_c_fe,
    tau_f_fe_impute = tau_f_fe_impute, tau_c_fe_impute = tau_c_fe_impute,
    tau_f_fe_se = tau_f_fe_se, tau_c_fe_se = tau_c_fe_se,
    tau_f_fe_impute_se = tau_f_fe_impute_se, tau_c_fe_impute_se = tau_c_fe_impute_se
  )
}
''')

