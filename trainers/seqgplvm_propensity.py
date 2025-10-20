import torch
from torch.distributions import Normal
import gpytorch 
from utils.training import load_train_cfg_from_json, materialize_cfg
from utils.checkpoints import latest_checkpoint_path, load_ckpt_any,train_dir, train_dir
from utils.propensity import propensity_dir
from utils.preprocessings import get_training_tensors
import pandas as pd
from utils.pathing import as_path
from pathlib import Path
from models.SeqGPLVM import SeqGPLVM, SeqGPLVMVal
import time,  os, json 



def propensity_seqgplvm(train_id: str,
                        pid_col: str = "patient_id",
                        time_col: str = "t",
                        treatment_col: str = "D",
                        covariate_cols_prefix: str = "x",
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                        sample_num: int = 100, 
                        sample_count: int = 0,         # number of A samples to draw
                        sample_independent: bool = False,  # for Gaussian, sample factorized N(mu,var) instead of full MVN
                       ):
    """
    Validation fine-tuning: load a trained SeqGPLVM, attach validation latents, 
    freeze all base params, and optimize only Z_val on the validation split.
    """
    # 1) Build *both* splits so we can reconstruct the trained model shape
    final_root = Path(os.environ.get("FINAL_ROOT", "./results")).expanduser()
    train_out = train_dir(final_root, "seqgplvm", train_id)
    if not train_out.exists():
        raise FileNotFoundError(f"Parent train run not found: {train_out}")
    
    data_ref  = json.loads((train_out / "data_ref.json").read_text(encoding="utf-8"))
    train_conf = load_train_cfg_from_json((train_out / "config.json"))
    train_conf = materialize_cfg(train_conf, device)
    

    df = pd.read_parquet(as_path(data_ref["data_file"]) / "data.parquet")
    df_manifest = json.loads((as_path(data_ref["data_file"]) / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads(as_path(data_ref["split_file"]).read_text(encoding="utf-8"))

    X, A, id2row = get_training_tensors(
        df,
        id_col=pid_col, time_col=time_col,
        treatment_col=treatment_col,
        covariate_cols_prefix=covariate_cols_prefix,
        treatment_lag=train_conf["treatment_lag"],
    )
    
    # prefer "val_ids" if present; fall back to "test_ids"
    train_ids = split.get("train_ids", [])
    train_rows = [id2row[pid] for pid in train_ids if pid in id2row] 
    X_train = X[train_rows].to(device)
    A_train = A[train_rows].to(device)



    val_ids = split.get("val_ids", [])
    test_ids = split.get("test_ids", [])
    test_val_ids = val_ids + test_ids 
    if len(test_val_ids) == 0:
        raise ValueError("No val_ids or test_ids found in split file.")
    
    val_rows   = [id2row[pid] for pid in test_val_ids   if pid in id2row]

    X_val   = X[val_rows].to(device)
    A_val   = A[val_rows].to(device)

    ckpt_path = latest_checkpoint_path(train_out)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in parent run: {train_out}")
    
    # 2) Rebuild the *trained* base model with TRAIN shapes and load its weights and eval mode
    model_base = SeqGPLVM(Y = A_train, X_cov = X_train, latent_dim = train_conf["latent_dim"], 
                     n_inducing_x = train_conf["num_inducing"], n_inducing_hidden = train_conf["num_inducing_hidden"],
                     init_z=None,z_initializer=train_conf["z_initializer"],
                     uniform_halfwidth=train_conf.get("uniform_halfwidth", None),
                     prior_std=train_conf.get("prior_std", None),device=device,
                     lik= train_conf["treatment_model"],
                     learn_inducing_locations = train_conf["learn_inducing_locations"],
                     use_titsias=train_conf["use_titsias"]).to(device)


    payload_train = load_ckpt_any(ckpt_path, map_location=device)
    model_base.load_state_dict(payload_train["model_state"], strict=True)

    model_base.eval()
    for lik in model_base.likelihoods:
        lik.eval()
    
    extra = payload_train.get("extra")
    param_hist    = extra.get("param_hist")
    variational_mean_train = torch.as_tensor(param_hist["Z.q_mu"][-1], 
                                         device=device, dtype=X_train.dtype) if param_hist else None
    variational_log_sigma_train = torch.as_tensor(param_hist["Z.q_log_sigma"][-1], 
                                         device=device, dtype=X_train.dtype) if param_hist else None
    if variational_mean_train is None or variational_log_sigma_train is None:
        raise ValueError("No variational params found in parent training checkpoint.")


    # 3) Rebuild the validation model and eval mode
    val_out = train_dir(final_root, "seqgplvm_val", train_id)

    if not val_out.exists():
        raise FileNotFoundError(f"Validation run directory not found: {val_out}. Please run `train_seqgplvm_val` first to create it.")

    ckpt_path = latest_checkpoint_path(val_out)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in validation run: {val_out}")
    payload_val = load_ckpt_any(ckpt_path, map_location=device)
    extra = payload_val.get("extra")
    param_hist    = extra.get("param_hist")
    variational_mean_val = torch.as_tensor(param_hist["Z_val.q_mu"][-1], 
                                       device=device, dtype=X_train.dtype)
    variational_log_sigma_val = torch.as_tensor(param_hist["Z_val.q_log_sigma"][-1], 
                                        device=device, dtype=X_train.dtype)
    
    if variational_mean_val is None or variational_log_sigma_val is None:
        raise ValueError("No variational params found in validation training checkpoint.") 
    
    model_val = SeqGPLVMVal.from_trained(model_base, X_val=X_val, Y_val=A_val).to(device)
    model_val.load_state_dict(payload_val["model_state"], strict=True)

    model_val.eval()
    for lik in model_val.likelihoods:
        lik.eval()
    
    propensity_dir_out = propensity_dir(final_root, "seqgplvm", train_id)
    propensity_dir_out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # we will reconstruct the the posterior over Z and not usign the built in method validation_model.Z() which returns N_val samples each time.
        # To speed up the process we want to perfrom the prediction task in one run

        qZ_dist_train = Normal(loc=variational_mean_train, scale=torch.nn.functional.softplus(variational_log_sigma_train))
        qZ_dist_val = Normal(loc=variational_mean_val, scale=torch.nn.functional.softplus(variational_log_sigma_val))

        with gpytorch.settings.cholesky_jitter(1e-3):
            z_star_train = qZ_dist_train.sample(torch.Size([sample_num]))
            z_star_val = qZ_dist_val.sample(torch.Size([sample_num]))
            prop_dic_train = model_base.propensity(X_cov_star = X_train, z_star = z_star_train , A_obs= A_train, z_integral= sample_num, sample_count= sample_count, sample_independent= sample_independent)
            prop_dic_val = model_val.propensity(X_cov_star = X_val, z_star = z_star_val , A_obs= A_val, z_integral= sample_num, sample_count= sample_count, sample_independent= sample_independent)

    # --- reassemble into global tensors ----------------------------------------
    K = sample_num
    T = X_train.size(1)
    N_total = len(id2row)
    S = sample_count

    # make sure both dicts have log_gps
    def ensure_log_gps(d):
        if d.get("log_gps") is not None:
            return d["log_gps"]                         # (N*, T, K)
        raise ValueError("propensity dict has no 'log_gps' entry.")

    # detect likelihood once (binary vs continuous)
    is_bernoulli = isinstance(model_base.likelihoods[0], gpytorch.likelihoods.BernoulliLikelihood)

    loggps_train = ensure_log_gps(prop_dic_train).to(torch.float32).cpu()  # (Ntrain, T)
    loggps_val   = ensure_log_gps(prop_dic_val).to(torch.float32).cpu()  # (Nval,   T)

    loggps_samples_train = prop_dic_train.get("log_gps_samples_z_meaned")
    loggps_samples_val   = prop_dic_val.get("log_gps_samples_z_meaned")
    if loggps_samples_train is not None: loggps_samples_train = loggps_samples_train.to(torch.float32).cpu()
    if loggps_samples_val   is not None: loggps_samples_val   = loggps_samples_val.to(torch.float32).cpu()

    # allocate global tensors (NaN where not applicable)
    loggps_all = torch.full((N_total, T, K), float("nan"), dtype=torch.float32)
    loggps_samples_all = torch.full((S, N_total, T), float("nan"), dtype=torch.float32)
    if is_bernoulli:
        prop_all = torch.full((N_total, T, K), float("nan"), dtype=torch.float32)
    else:
        prop_all = None

    # scatter into original row positions
    loggps_all[train_rows, :, :] = loggps_train
    loggps_all[val_rows,   :, :] = loggps_val
    if loggps_samples_train is not None:
        loggps_samples_all[:, train_rows, :] = loggps_samples_train
    if loggps_samples_val is not None:
        loggps_samples_all[:, val_rows, :] = loggps_samples_val
    
    prop_train = prop_dic_train.get("propensity")
    prop_val   = prop_dic_val.get("propensity")
    if prop_train is not None: prop_train = prop_train.to(torch.float32).cpu()
    if prop_val   is not None: prop_val   = prop_val.to(torch.float32).cpu()
    
    if is_bernoulli:
        prop_all[train_rows, :, :] = prop_train
        prop_all[val_rows,   :, :] = prop_val
    
    if sample_count > 0:
        A_samp_train = prop_dic_train["A_samples"].cpu()   # (S, N_train, T, K)
        A_samp_val   = prop_dic_val["A_samples"].cpu()     # (S, N_val,   T, K)

        A_samples_all = torch.full((S, N_total, T, K), float("nan"), dtype=A_samp_train.dtype)
        A_samples_all[:, train_rows, :, :] = A_samp_train
        A_samples_all[:, val_rows,   :, :] = A_samp_val

    # an availability mask is handy downstream
    mask_all = ~torch.isnan(loggps_all)

    # --- save once ------------------------------------------------------------
    payload = {
        "log_gps": loggps_all,        # (N_total, T)
        "log_gps_samples_z_meaned": loggps_samples_all, # (S, N_total, T)
        "mask": mask_all,             # (N_total, T, K)
        "is_bernoulli": is_bernoulli,
        # keep propensities only for binary (for convenience)
        "propensity": prop_all,       # None or (N_total, T, K)
        # (optional) store the z samples you used for reproducibility:
        "z_star_train": z_star_train.cpu(),  # (K, Ntrain, Q)
        "z_star_val":   z_star_val.cpu(),    # (K, Nval,   Q)
        "meta": {
            "train_id": train_id,
            "K": K,
            "T": T,
            "N_total": N_total,
            "N_train": len(train_rows),
            "N_val":   len(val_rows),
            "pid_col": pid_col,
            "time_col": time_col,
            "treatment_col": treatment_col,
            "likelihood": type(model_base.likelihoods[0]).__name__,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        # indexing helpers
        "index": {
            "row_by_pid": id2row,         # dict(pid -> row)
            "train_rows": train_rows,
            "val_rows":   val_rows,
            "pids_train": [pid for pid in train_ids if pid in id2row],
            "pids_val":   [pid for pid in test_val_ids if pid in id2row],
        },
    }

    if sample_count > 0:
        payload["A_samples"] = A_samples_all  # (S, N_total, T, K)

    out_path = propensity_dir_out / f"loggps_{train_id}.pt"
    torch.save(payload, out_path)
    print(f"Saved log–GPS tensor to: {out_path}  shape={tuple(loggps_all.shape)}")