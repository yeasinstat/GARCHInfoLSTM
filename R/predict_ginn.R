# ============================================================
# GINN -- Out-of-Sample Forecasting: predictGINN + AccuracyGINN
# ============================================================
#' @importFrom utils tail
NULL
#' Forecast h steps ahead from a fitted GINN model
#'
#' Forecasts future conditional variance h steps ahead using
#' recursive one-step prediction on variance sequences.
#'
#' @param fit     A GINN object returned by GINN().
#' @param h       Number of steps to forecast. Default 1.
#' @param lambda  Lambda value to use. NULL = best (lowest Train RMSE).
#' @param newdata Optional numeric vector of new DATA that arrived after
#'                training. Used to update the starting variance sequence.
#'
#' @return Data frame with columns: h, time_idx, forecast_var.
#' @export
predictGINN <- function(fit, h = 1, lambda = NULL, newdata = NULL) {

  # -- Select model -----------------------------------------
  nm <- if (is.null(lambda)) fit$meta$best_lam_nm else {
    if (lambda == 0.0) "standard" else paste0("lambda_", lambda)
  }
  if (!(nm %in% names(fit$results)))
    stop("lambda not found. Available: ",
         paste(sapply(fit$results, function(r) r$lambda), collapse=", "))

  model   <- fit$results[[nm]]$model
  scaler  <- fit$data_info$var_scaler
  seq_ln  <- fit$data_info$lag

  # -- Starting variance sequence ----------------------------
  cur_seq <- fit$data_info$last_var_seq_sc

  # If newdata (new prices) provided: compute their variance and update
  if (!is.null(newdata)) {
    nd  <- as.numeric(newdata)
    all_prices <- c(fit$data_info$prices, nd)
    all_ret    <- diff(all_prices) / all_prices[-length(all_prices)]
    new_ret    <- tail(all_ret, length(nd))
    mu_new     <- mean(new_ret)
    new_var    <- (new_ret - mu_new)^2
    new_var_sc <- scaler$scale(new_var)
    cur_seq    <- tail(c(fit$data_info$last_var_seq_sc, new_var_sc), seq_ln)
  }

  # -- Recursive h-step variance forecast -------------------
  forecasts_sc <- numeric(h)
  model$eval()
  for (t in seq_len(h)) {
    x_t <- torch::torch_tensor(
      array(cur_seq, dim=c(1L, seq_ln, 1L)),
      dtype=torch::torch_float()
    )
    pred_sc <- torch::with_no_grad({
      as.numeric(model(x_t)$detach()$reshape(list(1L)))
    })
    forecasts_sc[t] <- pred_sc
    cur_seq <- c(cur_seq[-1L], pred_sc)
  }

  # -- Inverse scale -- variance can't be negative ------------
  forecasts <- pmax(scaler$unscale(forecasts_sc), 0)


  n_new    <- if (!is.null(newdata)) length(newdata) else 0L
  last_idx <- fit$data_info$N_ret + n_new
  time_idx <- last_idx + seq_len(h)

  df <- data.frame(
    h            = seq_len(h),
    time_idx     = time_idx,
    forecast_var = round(forecasts, 10)
  )

  attr(df, "lambda") <- nm
  class(df) <- c("GINN_forecast", "data.frame")
  df
}


#' Accuracy of GINN forecasts vs actual values
#'
#' @param fit     A GINN object.
#' @param actual  Numeric vector of actual future PRICES after training.
#'                Their variance is computed and compared with forecasts.
#' @param lambda  Lambda to use. NULL = best.
#' @param verbose Logical. If \code{TRUE} (default), prints a formatted
#'                accuracy report to the console. Set to \code{FALSE} to
#'                suppress console output and only receive the returned
#'                data frame.
#'
#' @return Data frame with RMSE, MAE, R2 on variance.
#' @export
AccuracyGINN <- function(fit, actual, lambda = NULL, verbose = TRUE) {
  actual <- as.numeric(actual)
  h      <- length(actual)
  if (h < 2) stop("Need at least 2 actual values to compute variance.")

  # Forecast h steps
  fc   <- predictGINN(fit, h=h, lambda=lambda)
  pred_var <- fc$forecast_var

  # Compute actual variance from actual prices
  all_prices <- c(fit$data_info$prices, actual)
  all_ret    <- diff(all_prices) / all_prices[-length(all_prices)]
  new_ret    <- tail(all_ret, h)
  mu_new     <- mean(new_ret)
  actual_var <- (new_ret - mu_new)^2

  rmse  <- sqrt(mean((actual_var - pred_var)^2))
  mae   <- mean(abs(actual_var - pred_var))
  ss_r  <- sum((actual_var - pred_var)^2)
  ss_t  <- sum((actual_var - mean(actual_var))^2)
  r2    <- if (ss_t > 0) 1 - ss_r/ss_t else NA_real_
  denom <- abs(actual_var) + abs(pred_var); mask <- denom != 0
  smape <- if (any(mask)) mean(2*abs(actual_var[mask]-pred_var[mask])/denom[mask])*100 else 0

  lam_label <- gsub("lambda_","", attr(fc,"lambda"))

  if (verbose) {
    cat("\n=== AccuracyGINN \u2014 Out-of-Sample Accuracy (Variance) ===\n")
    cat(sprintf("  Lambda   : %s\n",  lam_label))
    cat(sprintf("  Horizon  : %d steps\n", h))
    cat(sprintf("  RMSE     : %.10f\n", rmse))
    cat(sprintf("  MAE      : %.10f\n", mae))
    cat(sprintf("  SMAPE    : %.4f%%\n", smape))
    cat("=========================================================\n\n")
  }

  df <- data.frame(Lambda=lam_label, h=h,
                   RMSE=round(rmse,10), MAE=round(mae,10),
                   SMAPE=round(smape,4))

  comparison <- data.frame(
    Step         = seq_len(h),
    Actual_Var   = round(actual_var,  10),
    Forecast_Var = round(pred_var,    10),
    Error        = round(actual_var - pred_var, 10),
    Abs_Error    = round(abs(actual_var - pred_var), 10)
  )
  attr(df, "comparison") <- comparison
  invisible(df)
}


#' @export
print.GINN_forecast <- function(x, ...) {
  lam <- gsub("lambda_","", attr(x,"lambda"))
  cat(sprintf("\n=== GINN Variance Forecast  |  lambda=%s  |  h=%d steps ===\n",
              lam, nrow(x)))
  print.data.frame(x, ...)
  cat("\n")
  invisible(x)
}
