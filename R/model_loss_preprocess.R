# ============================================================
# GINN -- Model, Loss, Preprocessing, Evaluation
# ============================================================
#' @importFrom stats approx
NULL
# -- LSTM Model -----------------------------------------------
.ginn_lstm <- function(hidden_size=32L, num_layers=1L, dropout=0.0) {
  hs   <- as.integer(hidden_size)
  nl   <- as.integer(num_layers)
  do_r <- if (nl > 1L) as.double(dropout) else 0.0

  GINNNet <- torch::nn_module(
    classname = "GINNNet",
    initialize = function(hs, nl, do_r) {
      self$lstm <- torch::nn_lstm(1L, hs, nl, batch_first=TRUE, dropout=do_r)
      self$fc   <- torch::nn_linear(hs, 1L)
    },
    forward = function(x) {
      out  <- self$lstm(x)[[1]]
      n    <- out$shape[2]
      last <- out$select(2L, n - 1L)   # 0-based index for last timestep
      self$fc(last)
    }
  )
  GINNNet(hs=hs, nl=nl, do_r=do_r)
}


# -- GINN Loss (Eq. 17) ---------------------------------------
# Loss = (1-lambda) x MSE(sigma2_gt,    sigma2_hat_LSTM)
#      + lambda     x MSE(sigma2_garch, sigma2_hat_LSTM)
# lambda=0.0 -> Standard LSTM (GT only -- no GARCH guidance)
# lambda=1.0 -> GINN-1 (GARCH only)
.ginn_loss <- function(y_pred, sigma2_gt, sigma2_garch, lambda) {
  mse_fn   <- torch::nn_mse_loss()
  loss_gt   <- mse_fn(y_pred, sigma2_gt)
  loss_garch<- mse_fn(y_pred, sigma2_garch)
  total     <- (1 - lambda) * loss_gt + lambda * loss_garch
  list(total = total, gt = loss_gt, garch = loss_garch)
}


# -- MinMax Scaler --------------------------------------------
.ginn_scaler <- function(all_values) {
  mn  <- min(all_values); mx <- max(all_values)
  rng <- max(mx - mn, 1e-12)
  list(
    scale   = function(x) (x - mn) / rng,
    unscale = function(x) x * rng + mn,
    min = mn, max = mx, range = rng
  )
}


# -- Sequence builder -----------------------------------------
.ginn_seqs <- function(series, seq_len) {
  n <- length(series)
  X <- matrix(NA_real_, nrow = n - seq_len, ncol = seq_len)
  y <- numeric(n - seq_len)
  for (i in seq_len(n - seq_len)) {
    X[i, ] <- series[i:(i + seq_len - 1L)]
    y[i]   <- series[i + seq_len]
  }
  list(X = X, y = y)
}


# -- 3-D tensor (batch, seq, 1) -------------------------------
.to3d <- function(mat) {
  arr <- array(mat, dim = c(nrow(mat), ncol(mat), 1L))
  torch::torch_tensor(arr, dtype = torch::torch_float())
}
.tocol <- function(vec) {
  torch::torch_tensor(matrix(vec, ncol = 1L), dtype = torch::torch_float())
}


# -- Full preprocessing ----------------------------------------
.ginn_preprocess <- function(sigma2_gt_all, sigma2_gt_train,
                              sigma2_gt_test, sigma2_garch_tr,
                              sigma2_garch_te, split, seq_len, batch_size) {

  n_train <- length(sigma2_gt_train)
  n_test  <- length(sigma2_gt_test)

  # Scale using all sigma^2_t so inverse-transform is consistent
  scaler     <- .ginn_scaler(sigma2_gt_all)
  sc_all     <- scaler$scale(sigma2_gt_all)

  # Scale GARCH with same scaler
  sc_garch_all <- scaler$scale(c(sigma2_garch_tr, sigma2_garch_te))

  # Supervised sequences on sigma^2_t
  seqs     <- .ginn_seqs(sc_all, seq_len)
  X_all    <- seqs$X; y_all <- seqs$y

  seq_split <- split - seq_len
  X_train   <- X_all[seq_len(seq_split), , drop = FALSE]
  y_train   <- y_all[seq_len(seq_split)]
  X_test    <- X_all[(seq_split + 1L):nrow(X_all), , drop = FALSE]
  y_test    <- y_all[(seq_split + 1L):length(y_all)]
  n_test_seq<- length(y_test)

  # GARCH aligned to sequences
  # Training: garch_sc[SEQ_LEN + i] aligns with y_train[i]
  garch_tr_aligned <- sc_garch_all[seq_len + seq_len(length(y_train))]
  garch_te_aligned <- sc_garch_all[(split + 1L):(split + n_test_seq)]

  # Internal val split (80% of training seqs)
  val_split <- floor(seq_split * 0.8)
  X_tr  <- X_train[seq_len(val_split), , drop = FALSE]; y_tr <- y_train[seq_len(val_split)]
  X_val <- X_train[(val_split + 1L):seq_split, , drop = FALSE]
  y_val <- y_train[(val_split + 1L):seq_split]
  garch_tr_guide <- garch_tr_aligned[seq_len(val_split)]

  # Torch tensors
  Xtr_t   <- .to3d(X_tr);   ytr_t  <- .tocol(y_tr)
  Xval_t  <- .to3d(X_val);  yval_t <- .tocol(y_val)
  Xte_t   <- .to3d(X_test); yte_t  <- .tocol(y_test)
  Xtr_full_t <- .to3d(X_train); ytr_full_t <- .tocol(y_train)
  guide_t    <- .tocol(garch_tr_guide)        # GARCH guidance for training
  garch_te_t <- .tocol(garch_te_aligned)      # GARCH guidance for test evaluation

  BATCH <- min(batch_size, nrow(X_tr))

  list(
    scaler = scaler, BATCH = BATCH,
    X_tr = X_tr, y_tr = y_tr,
    X_val = X_val, y_val = y_val,
    X_test = X_test, y_test = y_test,
    X_train = X_train, y_train = y_train,
    garch_tr_aligned = garch_tr_aligned,
    garch_te_aligned = garch_te_aligned,
    Xtr_t = Xtr_t, ytr_t = ytr_t,
    Xval_t = Xval_t, yval_t = yval_t,
    Xte_t = Xte_t, yte_t = yte_t,
    Xtr_full_t = Xtr_full_t, ytr_full_t = ytr_full_t,
    guide_t = guide_t, garch_te_t = garch_te_t,
    val_split = val_split, seq_split = seq_split,
    n_test_seq = n_test_seq
  )
}


# -- Evaluate variance predictions ----------------------------
.ginn_evaluate <- function(model, X_t, y_t, scaler) {
  model$eval()
  torch::with_no_grad({
    ps <- as.numeric(model(X_t)$detach()$reshape(list(-1L)))
  })
  preds   <- scaler$unscale(ps)
  actuals <- scaler$unscale(as.numeric(y_t$squeeze()))
  ss_res  <- sum((actuals - preds)^2)
  ss_tot  <- sum((actuals - mean(actuals))^2)
  r2      <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_
  denom <- abs(actuals) + abs(preds)
  mask  <- denom != 0
  smape <- if (any(mask)) mean(2*abs(actuals[mask]-preds[mask])/denom[mask])*100 else 0

  list(
    preds   = preds,
    actuals = actuals,
    rmse    = sqrt(mean((actuals - preds)^2)),
    mae     = mean(abs(actuals - preds)),
    smape   = smape
  )
}


# -- Full-data variance preprocessing (no test split) ----------
.ginn_preprocess_full <- function(sigma2_gt_all, sigma2_garch_all,
                                   seq_len, batch_size) {
  N_var <- length(sigma2_gt_all)
  scaler        <- .ginn_scaler(sigma2_gt_all)
  sc_all        <- scaler$scale(sigma2_gt_all)
  sc_garch_all  <- scaler$scale(sigma2_garch_all)

  seqs   <- .ginn_seqs(sc_all, seq_len)
  X_all  <- seqs$X; y_all <- seqs$y
  n_seqs <- nrow(X_all)

  # GARCH guidance aligned to sequences
  garch_guide <- sc_garch_all[seq_len + seq_len(n_seqs)]

  # Internal val split (last 20% of sequences)
  val_start <- floor(n_seqs * 0.8) + 1L
  val_split <- val_start - 1L
  X_tr  <- X_all[seq_len(val_split), , drop=FALSE]; y_tr <- y_all[seq_len(val_split)]
  X_val <- X_all[val_start:n_seqs, , drop=FALSE];   y_val <- y_all[val_start:n_seqs]
  guide_tr <- garch_guide[seq_len(val_split)]

  Xtr_t   <- .to3d(X_tr);   ytr_t   <- .tocol(y_tr)
  Xval_t  <- .to3d(X_val);  yval_t  <- .tocol(y_val)
  Xall_t  <- .to3d(X_all);  yall_t  <- .tocol(y_all)
  guide_t <- .tocol(as.numeric(guide_tr))

  BATCH <- min(batch_size, nrow(X_tr))

  # Last seq_len variance values for out-of-sample forecasting
  last_seq_scaled <- sc_all[(N_var - seq_len + 1L):N_var]

  list(
    scaler=scaler, BATCH=BATCH,
    X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
    X_all=X_all, y_all=y_all,
    val_split=val_split, n_seqs=n_seqs,
    Xtr_t=Xtr_t, ytr_t=ytr_t, Xval_t=Xval_t, yval_t=yval_t,
    Xall_t=Xall_t, yall_t=yall_t,
    guide_t=guide_t,
    last_seq_scaled=last_seq_scaled, seq_len=seq_len
  )
}
