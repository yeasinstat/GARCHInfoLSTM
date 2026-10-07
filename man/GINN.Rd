\name{GINN}
\alias{GINN}
\title{GARCH-Informed Neural Network (GINN) -- Volatility Forecasting}
\description{
Trains an LSTM on conditional variance sequences using the GINN composite loss:

\preformatted{
  Loss = (1-lambda) x MSE(sigma2_t, sigma2_hat_LSTM)
       + lambda     x MSE(sigma2_GARCH, sigma2_hat_LSTM)
}

Trains on FULL data -- no train/test split.
Best lambda selected by Training RMSE.
Use predictGINN() for out-of-sample volatility forecasting.

Pipeline:
\enumerate{
  \item Returns: \eqn{r_t = (y_t - y_{t-1}) / y_{t-1}}{r_t = (y_t - y_(t-1)) / y_(t-1)}
  \item AR(p) on full returns gives mean return mu_t
  \item GARCH(p,q) on full returns gives sigma2_GARCH
  \item Ground-truth variance: sigma2_t = (r_t - mu_t)^2
  \item LSTM trained on sigma2_t sequences with GINN loss
}
}
\usage{
GINN(
  data,
  mode = c("auto", "manual"),
  ar_max_lag = 5L,
  ar_lag = NULL,
  garch_p_grid = 1:3,
  garch_q_grid = 1:3,
  garch_mean_grid = c("constant", "ar", "arma"),
  garch_dist_grid = c("norm", "std", "ged", "snorm", "sstd", "sged"),
  garch_p = NULL,
  garch_q = NULL,
  garch_mean = NULL,
  garch_dist = NULL,
  lag,
  hidden_size = 32,
  num_layers = 1,
  lr = 0.001,
  dropout = 0.0,
  epochs = 500,
  patience = 30,
  tune_epochs = 50,
  batch_size = 16,
  lambda_list = seq(0.1, 0.9, 0.1),
  hidden_grid = c(8, 16, 24, 32, 40),
  layers_grid = c(1, 2),
  lr_grid = c(0.001, 0.003, 0.005),
  dropout_grid = c(0.0, 0.1, 0.2, 0.3),
  verbose = TRUE,
  seed = 42L
)
}
\arguments{
  \item{data}{Numeric data vector. ALL observations used for training.
    GINN internally computes returns and conditional variance.}

  \item{mode}{\code{"auto"} for automatic AR + GARCH + LSTM tuning.
    \code{"manual"} for user-supplied parameters.}

  \item{ar_max_lag}{Max AR lag to search (auto mode). Default 5.}

  \item{ar_lag}{Fixed AR lag (manual mode). NULL = auto.}

  \item{garch_p_grid}{ARCH order grid (auto mode). Default 1:3.}

  \item{garch_q_grid}{GARCH order grid (auto mode). Default 1:3.}

  \item{garch_mean_grid}{Mean model candidates for grid search:
    "constant" (intercept only), "ar" (AR(p) mean), "arma" (ARMA mean).
    Default c("constant", "ar", "arma").}

  \item{garch_dist_grid}{Error distribution candidates:
    "norm" (normal), "std" (Student-t), "ged" (GED),
    "snorm" (skew-normal), "sstd" (skew-t), "sged" (skew-GED).
    Default includes all six.}

  \item{garch_p}{Fixed ARCH order (manual mode).}

  \item{garch_q}{Fixed GARCH order (manual mode).}

  \item{garch_mean}{Fixed mean model (manual mode): "constant", "ar", or "arma".}

  \item{garch_dist}{Fixed distribution (manual mode).}

  \item{lag}{Variance sequence look-back length. COMPULSORY -- no default.
    User must specify. Example: lag = 5 uses the last 5 variance values
    to predict the next one.}

  \item{hidden_size}{LSTM hidden units (manual mode). Default 32.}

  \item{num_layers}{Stacked LSTM layers (manual mode). Default 1.}

  \item{lr}{Learning rate (manual mode). Default 0.001.}

  \item{dropout}{Dropout rate (manual mode). Default 0.0.}

  \item{epochs}{Max training epochs. Default 500.}

  \item{patience}{Early-stopping patience. Default 30.}

  \item{tune_epochs}{Epochs per grid-search combo (auto mode). Default 50.}

  \item{batch_size}{Mini-batch size. Default 16.}

  \item{lambda_list}{Lambda values to test. Default seq(0.1, 0.9, 0.1).
    lambda = 0.0: Standard LSTM (GT variance only, no GARCH guidance).
    lambda = 0.5: Balanced GINN (equal GT and GARCH weight).
    lambda = 1.0: GINN-1 (GARCH variance only).}

  \item{hidden_grid}{Hidden-unit candidates (auto mode).}

  \item{layers_grid}{Layer-count candidates (auto mode).}

  \item{lr_grid}{Learning-rate candidates (auto mode).}

  \item{dropout_grid}{Dropout candidates (auto mode).}

  \item{verbose}{Print progress messages. Default TRUE.}

  \item{seed}{Random seed. Default 42.}
}
\value{
S3 object of class \code{"GINN"} with elements:
\describe{
  \item{results}{Named list per lambda. Each entry contains: tr_rmse, tr_mae,
    tr_r2, tr_preds, tr_actuals (all in VARIANCE units), val_rmse,
    loss_hist, time, lambda, model.}
  \item{best_model}{Complete best lambda summary: lambda, hyperparams,
    ar_lag, garch_config, metrics table, predictions (Actual_Var vs
    Predicted_Var data frame), loss_history, model object, train_time.}
  \item{garch}{GARCH config, sigma2_all (full in-sample variance), coefs.}
  \item{ar}{AR lag and fitted mean returns mu_all.}
  \item{returns}{Full return series r_t.}
  \item{variance}{Ground-truth variance sigma2_t (all observations).}
  \item{best_hp}{Best LSTM hyperparameters found.}
  \item{meta}{Settings: lag, lambda_list, mode, best_lam_nm,
    selection = "Train RMSE".}
  \item{data_info}{prices, N, N_ret, var_scaler, last_var_seq_sc,
    lag, mu_all.}
}
}
\examples{
\donttest{
prices <- cumprod(1 + rnorm(60, 0.001, 0.02)) * 100

# -- Fast manual mode (fixed GARCH order + small LSTM) -------
fit <- GINN(prices, mode = "manual", lag = 5,
            ar_lag = 1, garch_p = 1, garch_q = 1,
            garch_mean = "constant", garch_dist = "norm",
            hidden_size = 4, num_layers = 1, lr = 0.01,
            epochs = 20, patience = 5,
            lambda_list = c(0.5), verbose = FALSE)
print(fit)
summary(fit)

# -- Forecast 12 steps of conditional variance --------------
fc <- predictGINN(fit, h = 12)
print(fc)

# Volatility = sqrt(variance)
sqrt(fc$forecast_var)

# -- Diagnostic plots ---------------------------------------
plotGINN(fit, h = 12)      # actual + predicted + forecast
plotlambda(fit)             # Training RMSE by lambda

# -- Best model details -------------------------------------
fit$best_model
fit$best_model$lambda
fit$best_model$hyperparams
fit$best_model$metrics
fit$best_model$predictions   # Actual_Var vs Predicted_Var

# -- Manual mode with a couple of lambdas --------------------
fit2 <- GINN(
  prices,
  mode       = "manual",
  lag        = 5,
  ar_lag     = 1,
  garch_p    = 1,
  garch_q    = 1,
  garch_mean = "ar",
  garch_dist = "norm",
  hidden_size = 4,
  num_layers  = 1,
  lr          = 0.01,
  epochs      = 20,
  patience    = 5,
  dropout     = 0.0,
  lambda_list = c(0.3, 0.7),
  verbose     = FALSE
)
summary(fit2)

# -- Accuracy vs actual future prices ----------------------
new_prices <- tail(prices, 1) * cumprod(1 + rnorm(12, 0.001, 0.02))
AccuracyGINN(fit, actual = new_prices)

# Plot with actual future variance overlaid in green
plotGINN(fit, h = 12, actual = new_prices)
}
}
