# ============================================================
# GINN -- GARCH-Informed Neural Network  |  Main Entry Point
# Trains on FULL data. Use predictGINN() for out-of-sample forecast.
# Best lambda selected by TRAINING RMSE.
# ============================================================

.fmt_time_ginn <- function(t) {
  if      (t < 60)   paste0(round(t,    2), " seconds")
  else if (t < 3600) paste0(round(t/60, 2), " minutes")
  else               paste0(round(t/3600,2), " hours")
}
#' @export
GINN <- function(
  data,
  mode            = c("auto", "manual"),
  ar_max_lag      = 5L,
  ar_lag          = NULL,
  garch_p_grid    = 1:3,
  garch_q_grid    = 1:3,
  garch_mean_grid = c("constant", "ar", "arma"),
  garch_dist_grid = c("norm", "std", "ged", "snorm", "sstd", "sged"),
  garch_p         = NULL,
  garch_q         = NULL,
  garch_mean      = NULL,
  garch_dist      = NULL,
  lag,                      # compulsory -- user must specify
  hidden_size     = 32,
  num_layers      = 1,
  lr              = 0.001,
  dropout         = 0.0,
  epochs          = 500,
  patience        = 30,
  tune_epochs     = 50,
  batch_size      = 16,
  lambda_list     = seq(0.1, 0.9, 0.1),
  hidden_grid     = c(8, 16, 24, 32, 40),
  layers_grid     = c(1, 2),
  lr_grid         = c(0.001, 0.003, 0.005),
  dropout_grid    = c(0.0, 0.1, 0.2, 0.3),
  verbose         = TRUE,
  seed            = 42L
) {
  mode <- match.arg(mode)
  data <- as.numeric(data)
  if (any(is.na(data))) {
    if (verbose) cli::cli_alert_info("Missing values \u2014 interpolating.")
    data <- .ginn_interp(data)
  }
  N <- length(data)
  if (N < 30) stop("GINN requires at least 30 observations.")
  seq_len <- as.integer(lag)
  if (seq_len < 1L) stop("lag must be a positive integer (e.g. lag = 5).")
  set.seed(seed); torch::torch_manual_seed(seed)

  if (verbose) {
    cli::cli_h1("GINN \u2014 GARCH-Informed Neural Network")
    cli::cli_alert_info("Mode       : {toupper(mode)}")
    cli::cli_alert_info("N (prices) : {N}  |  lag={lag}  |  Training on FULL data")
    cli::cli_alert_info("Lambda     : [{paste(lambda_list, collapse=', ')}]")
    cli::cli_alert_info("Best lambda: selected by Training RMSE")
  }

  # -- STEP 1: Returns on full data --------------------------
  returns <- diff(data) / data[-N]
  N_ret   <- length(returns)
  if (verbose) cli::cli_alert_info("[STEP 1] Returns: N={N_ret}  Mean={round(mean(returns),6)}  Std={round(sd(returns),6)}")

  # -- STEP 2: AR on full returns ----------------------------
  ar_res <- .fit_ar_full(returns, max_lag=ar_max_lag, fixed_lag=ar_lag, verbose=verbose)

  # -- STEP 3: GARCH on full returns -------------------------
  garch_res <- .fit_garch_full(
    returns     = returns,
    ar_lag      = ar_res$lag,
    mode        = mode,
    p_grid      = garch_p_grid,  q_grid = garch_q_grid,
    mean_grid   = garch_mean_grid, dist_grid = garch_dist_grid,
    fixed_p     = garch_p,  fixed_q    = garch_q,
    fixed_mean  = garch_mean, fixed_dist = garch_dist,
    verbose     = verbose
  )

  # -- STEP 4: Ground-truth variance ------------------------
  mu_all          <- ar_res$mu_all
  sigma2_gt_all   <- (returns - mu_all)^2
  sigma2_garch_all<- garch_res$sigma2_all

  if (verbose) {
    cli::cli_alert_info("[STEP 4] sigma2_t: Mean={round(mean(sigma2_gt_all),8)}  Std={round(sd(sigma2_gt_all),8)}")
  }

  # -- STEP 5: Preprocess full variance data -----------------
  pp <- .ginn_preprocess_full(
    sigma2_gt_all    = sigma2_gt_all,
    sigma2_garch_all = sigma2_garch_all,
    seq_len          = seq_len,
    batch_size       = batch_size
  )

  # -- STEP 6: Hyperparameter selection ---------------------
  if (mode == "auto") {
    if (verbose) cli::cli_h2("Grid Search \u2014 LSTM Hyperparameters")
    tune_out <- .ginn_tune(pp=pp, hidden_grid=hidden_grid, layers_grid=layers_grid,
                            lr_grid=lr_grid, dropout_grid=dropout_grid,
                            tune_epochs=tune_epochs, batch_size=batch_size,
                            seed=seed, verbose=verbose)
    best_hp    <- tune_out$best
    tuning_log <- tune_out$log
  } else {
    best_hp    <- list(hidden=hidden_size, layers=num_layers, lr=lr, dropout=dropout)
    tuning_log <- NULL
    if (verbose) {
      cli::cli_h2("Manual Hyperparameters")
      cli::cli_alert_info("hidden={best_hp$hidden}  layers={best_hp$layers}  lr={best_hp$lr}  dropout={best_hp$dropout}")
    }
  }

  # -- STEP 7: Training -- Standard + each lambda -------------
  if (verbose) cli::cli_h2("Training on Full Data  (Standard(lambda=0) + {length(lambda_list)} lambda values)")

  all_lams  <- c(0.0, lambda_list)
  lam_names <- ifelse(all_lams==0.0, "standard", paste0("lambda_", all_lams))
  results   <- vector("list", length(all_lams))
  names(results) <- lam_names

  for (i in seq_along(all_lams)) {
    lam   <- all_lams[i]
    label <- if (lam==0.0) "Standard (lambda=0.0)" else paste0("GINN (lambda=",lam,")")
    if (verbose) cli::cli_alert_info("[LSTM] {label}")

    set.seed(seed); torch::torch_manual_seed(seed)
    res <- .ginn_train_full(pp=pp, best_hp=best_hp, lam=lam,
                             epochs=epochs, patience=patience,
                             batch_size=batch_size, verbose=verbose)
    results[[i]] <- res
    if (verbose)
      cli::cli_alert_success(
        "  Train RMSE={round(res$tr_rmse,8)}  Val RMSE={round(res$val_rmse,8)}  SMAPE={round(res$tr_smape,2)}%"
      )
  }

  # -- Best lambda by TRAIN RMSE -----------------------------
  tr_rmses    <- sapply(results, function(r) r$tr_rmse)
  best_lam_nm <- names(which.min(tr_rmses))

  if (verbose)
    cli::cli_alert_success("Best lambda (Train RMSE): {gsub('lambda_','',best_lam_nm)}  RMSE={round(min(tr_rmses),8)}")

  # -- Build best_model summary ------------------------------
  bm <- results[[best_lam_nm]]
  best_model <- list(
    lambda      = bm$lambda,
    lambda_name = best_lam_nm,
    hyperparams = list(
      hidden_size = best_hp$hidden,
      num_layers  = best_hp$layers,
      lr          = best_hp$lr,
      dropout     = best_hp$dropout,
      lag         = lag,
      epochs_run  = length(bm$loss_hist$total)
    ),
    ar_lag      = ar_res$lag,
    garch_config= garch_res$config,
    metrics     = data.frame(
      Metric = c("RMSE", "MAE", "SMAPE%"),
      Value  = c(round(bm$tr_rmse,  8),
                 round(bm$tr_mae,   8),
                 round(bm$tr_smape, 4)),
      stringsAsFactors=FALSE
    ),
    
    predictions = data.frame(
      Actual_Var    = round(bm$tr_actuals, 10),
      Predicted_Var = round(bm$tr_preds,   10)
    ),
    loss_history = data.frame(
      Total_Loss        = round(bm$loss_hist$total, 8),
      LSTM_Loss         = round(bm$loss_hist$gt,    8),
      GARCH_Guided_Loss = round(bm$loss_hist$garch, 8)
    ),
    model      = bm$model,
    train_time = .fmt_time_ginn(bm$time)
  )

  out <- structure(
    list(
      results    = results,
      best_model = best_model,
      garch      = garch_res,
      ar         = ar_res,
      returns    = returns,
      variance   = list(all=sigma2_gt_all),
      best_hp    = best_hp,
      tuning_log = tuning_log,
      meta = list(
        mode        = mode,
        lag         = lag,
        lambda_list = lambda_list,
        all_lams    = all_lams,
        epochs      = epochs,
        patience    = patience,
        tune_epochs = tune_epochs,
        batch_size  = batch_size,
        seed        = seed,
        best_lam_nm = best_lam_nm,
        selection   = "Train RMSE"
      ),
      data_info = list(
        prices           = data,
        N                = N,
        N_ret            = N_ret,
        var_scaler       = pp$scaler,
        last_var_seq_sc  = pp$last_seq_scaled,
        lag              = lag,
        mu_all           = mu_all
      )
    ),
    class = "GINN"
  )

  if (verbose) {
    cli::cli_alert_success("GINN complete.")
    cli::cli_alert_info("Best model : fit$best_model")
    cli::cli_alert_info("Forecast   : predictGINN(fit, h=12)")
  }
  out
}
