# ============================================================
# GINN -- AR and GARCH Fitting
# ============================================================
#' @importFrom stats AIC approx
#' @importFrom utils tail
NULL
# -- Interpolate NAs ------------------------------------------
.ginn_interp <- function(x) {
  idx <- which(!is.na(x))
  if (length(idx) == length(x)) return(x)
  approx(idx, x[idx], xout = seq_along(x), rule = 2)$y
}

# -- AR Auto-tune ---------------------------------------------
# Loops AR(1) ... AR(max_lag), selects by AIC
.fit_ar <- function(train_ret, test_ret, max_lag = 5L,
                    fixed_lag = NULL, verbose = TRUE) {

  if (verbose) cli::cli_alert_info("[AR AUTO-TUNE] Searching best AR(p) ...")
  max_lag <- min(max_lag, floor(length(train_ret) / 5L))

  if (!is.null(fixed_lag)) {
    best_lag <- fixed_lag
    if (verbose) cli::cli_alert_info("[AR] Using fixed lag = {best_lag}")
  } else {
    best_aic <- Inf; best_lag <- 1L
    for (lag in seq_len(max_lag)) {
      tryCatch({
        fit <- stats::arima(train_ret, order = c(lag, 0L, 0L),
                            include.mean = TRUE, method = "ML")
        aic <- AIC(fit)
        if (verbose) cli::cli_alert_info("  AR({lag}) AIC={round(aic,4)}")
        if (aic < best_aic) { best_aic <- aic; best_lag <- lag }
      }, error = function(e) {
        if (verbose) cli::cli_alert_warning("  AR({lag}) failed")
      })
    }
    if (verbose) cli::cli_alert_success("[AR] Best: AR({best_lag})  AIC={round(best_aic,4)}")
  }

  # Refit best model
  best_fit <- stats::arima(train_ret, order = c(best_lag, 0L, 0L),
                           include.mean = TRUE, method = "ML")

  # In-sample fitted values (padded to match train length)
  fitted_raw <- as.numeric(stats::fitted(best_fit))
  pad_n      <- length(train_ret) - length(fitted_raw)
  mu_train   <- c(rep(fitted_raw[1], pad_n), fitted_raw)
  if (any(is.na(mu_train)))
    mu_train[is.na(mu_train)] <- mean(mu_train, na.rm=TRUE)
  # Out-of-sample forecast
  fc       <- stats::predict(best_fit, n.ahead = length(test_ret))
  mu_test  <- as.numeric(fc$pred)

  list(
    lag      = best_lag,
    fit      = best_fit,
    mu_train = mu_train,
    mu_test  = mu_test
  )
}


# -- GARCH Auto-tune (using rugarch) --------------------------
.fit_garch <- function(train_ret, test_ret, ar_lag,
                       mode = "auto",
                       p_grid = 1:3, q_grid = 1:3,
                       mean_grid = c("constant", "ar", "arma"),
                       dist_grid = c("norm","std","ged","snorm","sstd","sged"),
                       fixed_p = NULL, fixed_q = NULL,
                       fixed_mean = NULL, fixed_dist = NULL,
                       verbose = TRUE) {

  scaled_train <- train_ret * 100   # GARCH convention: rescale

  if (mode == "manual" && !is.null(fixed_p)) {
    # Manual: use user-supplied config
    best_p    <- fixed_p
    best_q    <- fixed_q
    best_mean <- fixed_mean %||% "ar"
    best_dist <- fixed_dist %||% "norm"
    tune_log  <- NULL
    if (verbose) {
      cli::cli_alert_info("[GARCH] Manual: GARCH({best_p},{best_q}) mean={best_mean} dist={best_dist}")
    }
  } else {
    # Auto: grid search over (p, q, mean, dist) \u2014 select by AIC
    if (verbose) cli::cli_alert_info("[GARCH AUTO-TUNE] Searching (p,q,mean,dist) ...")
    best_aic  <- Inf
    best_p    <- 1L; best_q <- 1L
    best_mean <- "ar"; best_dist <- "norm"
    tune_rows <- list()

    for (p in p_grid) {
      for (q in q_grid) {
        for (mn in mean_grid) {
          for (ds in dist_grid) {
            tryCatch({
              arma_ord <- switch(mn,
                                 "zero"     = c(0L, 0L),
                                 "constant" = c(0L, 0L),
                                 "ar"       = c(ar_lag, 0L),
                                 "arma"     = c(ar_lag, 1L),
                                 c(0L, 0L)
              )
              inc_mean <- mn != "zero"
              spec <- rugarch::ugarchspec(
                variance.model   = list(model = "sGARCH",
                                        garchOrder = c(p, q)),
                mean.model       = list(armaOrder = arma_ord,
                                        include.mean = inc_mean),
                distribution.model = ds
              )
              fit <- rugarch::ugarchfit(spec, scaled_train,
                                       solver = "hybrid", out.sample = 0)
              aic_val <- rugarch::infocriteria(fit)[1]   # AIC
              if (verbose) {
                cli::cli_alert_info("  GARCH({p},{q}) mean={mn} dist={ds}  AIC={round(aic_val,3)}")
              }
              tune_rows[[length(tune_rows)+1]] <- list(
                p=p, q=q, mean=mn, dist=ds, aic=aic_val
              )
              if (aic_val < best_aic) {
                best_aic  <- aic_val
                best_p    <- p; best_q <- q
                best_mean <- mn; best_dist <- ds
              }
            }, error = function(e) {
              if (verbose)
                cli::cli_alert_warning("  GARCH({p},{q}) mean={mn} dist={ds} FAILED")
            })
          }
        }
      }
    }
    tune_log <- do.call(rbind, lapply(tune_rows, as.data.frame))
    if (verbose) {
      cli::cli_alert_success(
        "[GARCH Best] GARCH({best_p},{best_q}) mean={best_mean} dist={best_dist}  AIC={round(best_aic,3)}"
      )
    }
  }

  # -- Fit best GARCH model ----------------------------------
  t0 <- proc.time()["elapsed"]
  arma_ord <- switch(best_mean,
                     "zero"     = c(0L, 0L),
                     "constant" = c(0L, 0L),
                     "ar"       = c(ar_lag, 0L),
                     "arma"     = c(ar_lag, 1L),
                     c(0L, 0L)
  )
  inc_mean <- best_mean != "zero"
  spec_best <- rugarch::ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = c(best_p, best_q)),
    mean.model         = list(armaOrder = arma_ord, include.mean = inc_mean),
    distribution.model = best_dist
  )
  garch_fit <- rugarch::ugarchfit(spec_best, scaled_train,
                                  solver = "hybrid", out.sample = 0)
  t_fit <- proc.time()["elapsed"] - t0

  # In-sample conditional variance (/ 100^2 to undo scaling)
  sigma2_train <- as.numeric(rugarch::sigma(garch_fit))^2 / 1e4

  # Out-of-sample forecast
  fc_garch     <- rugarch::ugarchforecast(garch_fit,
                                          n.ahead = length(test_ret))
  sigma2_test  <- as.numeric(rugarch::sigma(fc_garch))^2 / 1e4

  # Align lengths
  sigma2_train <- sigma2_train[seq_len(length(train_ret))]
  sigma2_test  <- sigma2_test[seq_len(length(test_ret))]

  # Metrics vs realized variance
  gt_test  <- (test_ret - 0)^2    # simple realized variance for GARCH standalone
  rmse_g   <- sqrt(mean((gt_test - sigma2_test)^2))
  mae_g    <- mean(abs(gt_test - sigma2_test))

  params_g <- length(rugarch::coef(garch_fit))
  if (verbose) {
    cli::cli_alert_success(
      "[GARCH] Fit time={round(t_fit,2)}s  Params={params_g}  RMSE(vs r^2_t)={round(rmse_g,8)}"
    )
  }

  list(
    config      = list(p=best_p, q=best_q, mean=best_mean, dist=best_dist),
    fit         = garch_fit,
    sigma2_train= sigma2_train,
    sigma2_test = sigma2_test,
    rmse        = rmse_g,
    mae         = mae_g,
    n_params    = params_g,
    time        = t_fit,
    tune_log    = if (exists("tune_log")) tune_log else NULL,
    coefs       = as.list(rugarch::coef(garch_fit))
  )
}

# -- AR on FULL returns -----------------------------------------
.fit_ar_full <- function(returns, max_lag=5L, fixed_lag=NULL, verbose=TRUE) {
  max_lag <- min(max_lag, floor(length(returns)/5L))
  if (!is.null(fixed_lag)) {
    best_lag <- fixed_lag
    if (verbose) cli::cli_alert_info("[AR] Fixed lag = {best_lag}")
  } else {
    if (verbose) cli::cli_alert_info("[AR AUTO] Searching best AR(p) on full returns ...")
    best_aic <- Inf; best_lag <- 1L
    for (lag in seq_len(max_lag)) {
      tryCatch({
        fit <- stats::arima(returns, order=c(lag,0L,0L), include.mean=TRUE, method="ML")
        aic <- AIC(fit)
        if (verbose) cli::cli_alert_info("  AR({lag}) AIC={round(aic,4)}")
        if (aic < best_aic) { best_aic <- aic; best_lag <- lag }
      }, error=function(e) NULL)
    }
    if (verbose) cli::cli_alert_success("[AR] Best: AR({best_lag})")
  }
  best_fit   <- stats::arima(returns, order=c(best_lag,0L,0L), include.mean=TRUE, method="ML")
  fitted_raw <- as.numeric(stats::fitted(best_fit))
  pad_n      <- length(returns) - length(fitted_raw)
  mu_all     <- c(rep(fitted_raw[1], pad_n), fitted_raw)

  # Replace any NA with unconditional mean (handles initial AR values)
  if (any(is.na(mu_all))) {
    mu_mean <- mean(mu_all, na.rm = TRUE)
    if (is.nan(mu_mean) || is.na(mu_mean)) mu_mean <- mean(returns, na.rm = TRUE)
    mu_all[is.na(mu_all)] <- mu_mean
  }

  list(lag=best_lag, fit=best_fit, mu_all=mu_all)
}

# -- GARCH on FULL returns --------------------------------------
.fit_garch_full <- function(returns, ar_lag, mode="auto",
                            p_grid=1:3, q_grid=1:3,
                            mean_grid=c("constant", "ar", "arma"),
                            dist_grid=c("norm","std","ged","snorm","sstd","sged"),
                             fixed_p=NULL, fixed_q=NULL,
                             fixed_mean=NULL, fixed_dist=NULL, verbose=TRUE) {
  scaled <- returns * 100
  N_ret  <- length(returns)

  if (mode=="manual" && !is.null(fixed_p)) {
    best_p <- fixed_p; best_q <- fixed_q
    best_mean <- fixed_mean %||% "ar"; best_dist <- fixed_dist %||% "norm"
    if (verbose) cli::cli_alert_info("[GARCH] Manual: GARCH({best_p},{best_q}) mean={best_mean} dist={best_dist}")
  } else {
    if (verbose) cli::cli_alert_info("[GARCH AUTO] Grid search on full returns ...")
    best_aic <- Inf; best_p <- 1L; best_q <- 1L
    best_mean <- "arma"; best_dist <- "norm"
    for (p in p_grid) for (q in q_grid) for (mn in mean_grid) for (ds in dist_grid) {
      tryCatch({
        arma_ord <- switch(mn,
                           "zero"     = c(0L, 0L),
                           "constant" = c(0L, 0L),
                           "ar"       = c(ar_lag, 0L),
                           "arma"     = c(ar_lag, 1L),
                           c(0L, 0L)
        )
        inc_mean <- mn != "zero"
        spec <- rugarch::ugarchspec(
          variance.model=list(model="sGARCH", garchOrder=c(p,q)),
          mean.model=list(armaOrder=arma_ord, include.mean=(mn!="zero")),
          distribution.model=ds)
        fit  <- rugarch::ugarchfit(spec, scaled, solver="hybrid", out.sample=0)
        aic_val <- rugarch::infocriteria(fit)[1]
        if (verbose) cli::cli_alert_info("  GARCH({p},{q}) mean={mn} dist={ds}  AIC={round(aic_val,3)}")
        if (aic_val < best_aic) {
          best_aic <- aic_val; best_p <- p; best_q <- q
          best_mean <- mn; best_dist <- ds
        }
      }, error=function(e) NULL)
    }
    if (verbose) cli::cli_alert_success("[GARCH Best] GARCH({best_p},{best_q}) mean={best_mean} dist={best_dist}")
  }

  t0 <- proc.time()["elapsed"]
  arma_ord <- switch(best_mean,
                     "zero"     = c(0L, 0L),
                     "constant" = c(0L, 0L),
                     "ar"       = c(ar_lag, 0L),
                     "arma"     = c(ar_lag, 1L),
                     c(0L, 0L)
  )
  spec_best <- rugarch::ugarchspec(
    variance.model=list(model="sGARCH", garchOrder=c(best_p,best_q)),
    mean.model=list(armaOrder=arma_ord, include.mean=(best_mean!="zero")),
    distribution.model=best_dist)
  garch_fit <- rugarch::ugarchfit(spec_best, scaled, solver="hybrid", out.sample=0)
  t_fit     <- proc.time()["elapsed"] - t0

  # In-sample sigma2 for ALL returns
  sigma2_all <- as.numeric(rugarch::sigma(garch_fit))^2 / 1e4

  if (verbose)
    cli::cli_alert_success("[GARCH] Fit time={round(t_fit,2)}s  Params={length(rugarch::coef(garch_fit))}")

  list(
    config      = list(p=best_p, q=best_q, mean=best_mean, dist=best_dist),
    fit         = garch_fit,
    sigma2_all  = sigma2_all,
    n_params    = length(rugarch::coef(garch_fit)),
    time        = t_fit,
    coefs       = as.list(rugarch::coef(garch_fit))
  )
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
