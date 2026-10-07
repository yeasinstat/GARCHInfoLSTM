# ============================================================
# GINN -- Hyperparameter Tuning + Training Loop
# ============================================================

# -- Grid Search ----------------------------------------------
.ginn_tune <- function(pp, hidden_grid, layers_grid, lr_grid,
                       dropout_grid, tune_epochs, batch_size, seed, verbose) {

  grid   <- expand.grid(hidden  = hidden_grid,
                        layers  = layers_grid,
                        lr      = lr_grid,
                        dropout = dropout_grid,
                        stringsAsFactors = FALSE)
  n_comb <- nrow(grid)
  if (verbose) cli::cli_alert_info("{n_comb} hyperparameter combinations to evaluate")

  dl  <- .ginn_dl(pp$Xtr_t, pp$ytr_t, pp$BATCH, shuffle = TRUE)
  log <- vector("list", n_comb)

  for (i in seq_len(n_comb)) {
    h <- grid$hidden[i];  l <- grid$layers[i]
    r <- grid$lr[i];      d <- grid$dropout[i]

    set.seed(seed); torch::torch_manual_seed(seed)
    m   <- .ginn_lstm(hidden_size = h, num_layers = l, dropout = d)
    opt <- torch::optim_adam(m$parameters, lr = r)
    m$train()

    for (ep in seq_len(tune_epochs)) {
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]
        opt$zero_grad()
        out <- .ginn_loss(m(bX), by, by, lambda = 0.0)   # GT only for tuning
        out$total$backward(); opt$step()
      })
    }

    ev   <- .ginn_evaluate(m, pp$Xval_t, pp$yval_t, pp$scaler)
    log[[i]] <- list(hidden=h, layers=l, lr=r, dropout=d, val_rmse=ev$rmse)
    if (verbose)
      cli::cli_alert_info("  [{i}/{n_comb}] h={h} l={l} lr={r} d={d}  Val RMSE={round(ev$rmse,10)}")
  }

  log_df <- do.call(rbind, lapply(log, as.data.frame))
  bi     <- which.min(log_df$val_rmse)
  best   <- as.list(log_df[bi, ])

  if (verbose) {
    cli::cli_alert_success(
      "Best HP: h={best$hidden}  l={best$layers}  lr={best$lr}  d={best$dropout}  Val RMSE={round(best$val_rmse,10)}"
    )
  }
  list(best = best, log = log_df)
}


# -- Train one GINN model -------------------------------------
.ginn_train <- function(pp, best_hp, lam, epochs, patience,
                        batch_size, verbose) {

  BATCH  <- min(batch_size, nrow(pp$X_tr))
  model  <- .ginn_lstm(hidden_size = best_hp$hidden,
                        num_layers  = best_hp$layers,
                        dropout     = best_hp$dropout)
  opt    <- torch::optim_adam(model$parameters, lr = best_hp$lr)
  mse_fn <- torch::nn_mse_loss()
  best_v <- Inf; counter <- 0L
  best_params <- lapply(model$parameters, function(p) p$clone()$detach())
  hist   <- list(total = numeric(), gt = numeric(),
                 garch = numeric(), val = numeric())
  t0     <- proc.time()["elapsed"]

  if (lam == 0.0) {
    # -- Standard (lambda=1): GT variance only ------------------
    dl <- .ginn_dl(pp$Xtr_t, pp$ytr_t, BATCH, shuffle = TRUE)
    for (epoch in seq_len(epochs)) {
      model$train(); ep <- 0.0
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]
        opt$zero_grad()
        out <- .ginn_loss(model(bX), by, by, lambda = 0.0)
        out$total$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, 1.0)
        opt$step(); ep <- ep + out$total$item()
      })
      ep <- ep / max(length(dl), 1L)
      hist$total <- c(hist$total, ep); hist$gt    <- c(hist$gt,    ep)
      hist$garch <- c(hist$garch, 0.0); hist$val <- c(hist$val, NA)

      vl <- torch::with_no_grad({
        mse_fn(model(pp$Xval_t), pp$yval_t)$item()
      })
      hist$val[length(hist$val)] <- vl

      if (vl < best_v) { best_v <- vl; counter <- 0L
        best_params <- lapply(model$parameters, function(p) p$clone()$detach()) }
      else counter <- counter + 1L
      if (counter >= patience) {
        if (verbose) cli::cli_alert_info("  Early stop @ epoch {epoch}"); break
      }
    }

  } else {
    # -- GINN (lambda<1): GT + GARCH guidance -------------------
    dl <- .ginn_dl_guided(pp$Xtr_t, pp$ytr_t, pp$guide_t, BATCH)
    for (epoch in seq_len(epochs)) {
      model$train(); et <- eg <- egr <- 0.0
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]; bg <- batch[[3]]
        opt$zero_grad()
        out <- .ginn_loss(model(bX), by, bg, lambda = lam)
        out$total$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, 1.0)
        opt$step()
        et  <- et  + out$total$item()
        eg  <- eg  + out$gt$item()
        egr <- egr + out$garch$item()
      })
      n <- max(length(dl), 1L)
      hist$total <- c(hist$total, et/n); hist$gt    <- c(hist$gt, eg/n)
      hist$garch <- c(hist$garch, egr/n); hist$val <- c(hist$val, NA)

      vl <- torch::with_no_grad({
        mse_fn(model(pp$Xval_t), pp$yval_t)$item()
      })
      hist$val[length(hist$val)] <- vl

      if (vl < best_v) { best_v <- vl; counter <- 0L
        best_params <- lapply(model$parameters, function(p) p$clone()$detach()) }
      else counter <- counter + 1L
      if (counter >= patience) {
        if (verbose) cli::cli_alert_info("  Early stop @ epoch {epoch}"); break
      }
    }
  }

  torch::with_no_grad({
    pnames <- names(model$parameters)
    for (i in seq_along(pnames))
      model$parameters[[pnames[i]]]$copy_(best_params[[i]])
  })
  ev    <- .ginn_evaluate(model, pp$Xte_t,      pp$yte_t,      pp$scaler)
  tr_ev <- .ginn_evaluate(model, pp$Xtr_full_t, pp$ytr_full_t, pp$scaler)
  elapsed <- proc.time()["elapsed"] - t0

  list(
    rmse       = ev$rmse,
    mae        = ev$mae,
    r2         = ev$r2,
    preds      = ev$preds,
    actuals    = ev$actuals,
    tr_rmse    = tr_ev$rmse,
    tr_mae     = tr_ev$mae,
    tr_r2      = tr_ev$r2,
    tr_preds   = tr_ev$preds,
    tr_actuals = tr_ev$actuals,
    loss_hist  = hist,
    time       = elapsed,
    lambda     = lam,
    n_params   = sum(sapply(model$parameters, function(p) p$numel()))
  )
}


# -- DataLoader helpers ----------------------------------------
.ginn_dl <- function(X_t, y_t, batch_size, shuffle = TRUE) {
  ds <- torch::tensor_dataset(X_t, y_t)
  torch::dataloader(ds, batch_size = batch_size,
                    shuffle = shuffle, drop_last = TRUE)
}

.ginn_dl_guided <- function(X_t, y_t, guide_t, batch_size) {
  bs <- min(batch_size, dim(X_t)[1])
  ds <- torch::tensor_dataset(X_t, y_t, guide_t)
  torch::dataloader(ds, batch_size = bs,
                    shuffle = FALSE, drop_last = TRUE)
}


# -- Train on FULL variance data (stores model) -----------------
.ginn_train_full <- function(pp, best_hp, lam, epochs, patience,
                              batch_size, verbose) {
  BATCH  <- min(batch_size, nrow(pp$X_tr))
  model  <- .ginn_lstm(hidden_size=best_hp$hidden, num_layers=best_hp$layers,
                        dropout=best_hp$dropout)
  opt    <- torch::optim_adam(model$parameters, lr=best_hp$lr)
  mse_fn <- torch::nn_mse_loss()
  best_v <- Inf; counter <- 0L
  best_params <- lapply(model$parameters, function(p) p$clone()$detach())
  hist <- list(total=numeric(), gt=numeric(), garch=numeric())
  t0   <- proc.time()["elapsed"]

  if (lam == 0.0) {
    dl <- .ginn_dl(pp$Xtr_t, pp$ytr_t, BATCH, shuffle=TRUE)
    for (epoch in seq_len(epochs)) {
      model$train(); ep <- 0.0
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]
        opt$zero_grad()
        out <- .ginn_loss(model(bX), by, by, lambda=0.0)
        out$total$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, 1.0)
        opt$step(); ep <- ep + out$total$item()
      })
      ep <- ep/max(length(dl),1L)
      hist$total <- c(hist$total,ep); hist$gt <- c(hist$gt,ep); hist$garch <- c(hist$garch,0.0)
      model$eval()
      vl <- torch::with_no_grad({ mse_fn(model(pp$Xval_t), pp$yval_t)$item() })
      if (vl < best_v) { best_v <- vl; counter <- 0L
        best_params <- lapply(model$parameters, function(p) p$clone()$detach()) }
      else counter <- counter + 1L
      if (counter >= patience) { if (verbose) cli::cli_alert_info("  Early stop @ epoch {epoch}"); break }
    }
  } else {
    dl <- .ginn_dl_guided(pp$Xtr_t, pp$ytr_t, pp$guide_t, BATCH)
    for (epoch in seq_len(epochs)) {
      model$train(); et <- eg <- egr <- 0.0
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]; bg <- batch[[3]]
        opt$zero_grad()
        out <- .ginn_loss(model(bX), by, bg, lambda=lam)
        out$total$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, 1.0)
        opt$step()
        et <- et+out$total$item(); eg <- eg+out$gt$item(); egr <- egr+out$garch$item()
      })
      n <- max(length(dl),1L)
      hist$total <- c(hist$total,et/n); hist$gt <- c(hist$gt,eg/n); hist$garch <- c(hist$garch,egr/n)
      model$eval()
      vl <- torch::with_no_grad({ mse_fn(model(pp$Xval_t), pp$yval_t)$item() })
      if (vl < best_v) { best_v <- vl; counter <- 0L
        best_params <- lapply(model$parameters, function(p) p$clone()$detach()) }
      else counter <- counter + 1L
      if (counter >= patience) { if (verbose) cli::cli_alert_info("  Early stop @ epoch {epoch}"); break }
    }
  }

  # Restore best weights
  torch::with_no_grad({
    pnames <- names(model$parameters)
    for (i in seq_along(pnames))
      model$parameters[[pnames[i]]]$copy_(best_params[[i]])
  })

  elapsed <- proc.time()["elapsed"] - t0
  ev      <- .ginn_evaluate(model, pp$Xall_t, pp$yall_t, pp$scaler)
  val_ev  <- .ginn_evaluate(model, pp$Xval_t, pp$yval_t, pp$scaler)

  list(
    tr_rmse    = ev$rmse,  tr_mae = ev$mae, tr_smape = ev$smape,
    tr_preds   = ev$preds, tr_actuals = ev$actuals,
    val_rmse   = val_ev$rmse,
    loss_hist  = hist, time = elapsed, lambda = lam,
    model      = model,
    n_params   = sum(sapply(model$parameters, function(p) p$numel()))
  )
}
