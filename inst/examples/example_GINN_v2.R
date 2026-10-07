# ============================================================
# GINN Package v2 — Example Usage
# GARCH-Informed Neural Network for VOLATILITY Forecasting
# Trains on FULL data. No train/test split.
# ============================================================
# Install dependencies:
#   install.packages(c("torch","rugarch","ggplot2","cli","coro"))
#   torch::install_torch()   # one-time ~500MB download
#
# Install GINN:
#   install.packages("path/to/GINN", repos=NULL, type="source")
# ============================================================

library(GINN)

# ── Simulate price series (or load your data) ─────────────────
set.seed(42)
prices <- cumprod(1 + rnorm(150, mean = 0.001, sd = 0.02)) * 100
# For real data:
# library(readxl)
# prices <- read_excel("Datasir.xlsx", sheet = "Onion")$Delhi


# ============================================================
# Example 1: AUTO mode (recommended)
# ============================================================
# Automatically selects: AR lag, GARCH(p,q,mean,dist), LSTM HP
# Trains on FULL price data — no train/test split
# Best lambda selected by Training RMSE

fit <- GINN(
  data        = prices,
  mode        = "auto",
  seq_len     = 5,
  lambda_list = seq(0.1, 0.9, 0.1),
  epochs      = 500,
  patience    = 30,
  tune_epochs = 50,
  verbose     = TRUE
)

# ── View results ───────────────────────────────────────────────
print(fit)          # compact table: Train RMSE, MAE, R2, Val RMSE per lambda
summary(fit)        # full metrics data frame
plot(fit)           # training loss history
plot(fit, "lambda") # Training RMSE bar chart

# ── Best model details ─────────────────────────────────────────
fit$best_model                      # complete summary
fit$best_model$lambda               # best lambda value
fit$best_model$hyperparams          # hidden, layers, lr, dropout, seq_len
fit$best_model$ar_lag               # AR lag used
fit$best_model$garch_config         # GARCH(p,q) config
fit$best_model$metrics              # RMSE, MAE, R2 table
fit$best_model$predictions          # Actual_Var vs Predicted_Var (in-sample)
fit$best_model$loss_history         # epoch-by-epoch loss
fit$best_model$train_time           # seconds

# ── Access components ──────────────────────────────────────────
fit$ar$lag               # best AR lag
fit$garch$config         # best GARCH config (p, q, mean, dist)
fit$garch$coefs          # GARCH coefficient estimates
fit$best_hp              # best LSTM hyperparameters
fit$tuning_log           # all grid-search combos
fit$variance$all         # full sigma2_t series (realized variance)
fit$returns              # full return series r_t

# ── Out-of-sample volatility forecast ─────────────────────────
fc <- predictGINN(fit, h = 12)        # forecast 12 steps ahead
print(fc)
# Columns: h, time_idx, forecast_var, lower, upper
# forecast_var = predicted conditional variance sigma2_t
# Volatility (std dev) = sqrt(forecast_var)
sqrt(fc$forecast_var)

# Forecast with specific lambda
fc5 <- predictGINN(fit, h = 6, lambda = 0.5)

# ── Plot: actual variance + predicted + forecast + CI ──────────
plotGINN(fit, h = 12)              # full plot
plotGINN(fit, h = 12, n_hist = 50) # show last 50 historical points only

# ── Lambda RMSE bar chart ──────────────────────────────────────
plotlambda(fit)    # gold = best lambda, red dashed = min RMSE

# ── When actual future prices arrive: accuracy check ──────────
new_prices <- tail(prices, 1) * cumprod(1 + rnorm(12, 0.001, 0.02))
acc <- AccuracyGINN(fit, actual = new_prices)
print(acc)
attr(acc, "comparison")   # step-by-step: Actual_Var vs Forecast_Var vs Error

# Plot with actual future variance overlaid in green
plotGINN(fit, h = 12, actual = new_prices)

# ── Training predictions (in-sample fit quality) ──────────────
train_df <- fit$best_model$predictions
head(train_df)
# Columns: Actual_Var, Predicted_Var, Error, Abs_Error


# ============================================================
# Example 2: MANUAL mode
# ============================================================
# User specifies ALL parameters directly

fit2 <- GINN(
  data       = prices,
  mode       = "manual",

  # AR
  ar_lag     = 1,

  # GARCH
  garch_p    = 1,
  garch_q    = 1,
  garch_mean = "zero",   # "zero" or "arma"
  garch_dist = "norm",   # "norm" (normal) or "std" (Student-t)

  # LSTM
  seq_len    = 5,
  hidden_size= 32,
  num_layers = 1,
  lr         = 0.001,
  dropout    = 0.0,

  # Lambda
  lambda_list = c(0.1, 0.3, 0.5, 0.7, 0.9),
  epochs      = 300,
  patience    = 30,
  verbose     = TRUE
)

print(fit2)
predictGINN(fit2, h = 12)
plotGINN(fit2, h = 12)
plotlambda(fit2)


# ============================================================
# Example 3: Real data loop (all markets)
# ============================================================
# library(readxl)
# df      <- read_excel("Datasir.xlsx", sheet = "Onion")
# markets <- setdiff(names(df), "Month")
#
# results_all <- lapply(markets, function(mkt) {
#   cat("\n>>> Market:", mkt, "\n")
#   GINN(data=df[[mkt]], mode="auto", seq_len=5, verbose=TRUE)
# })
# names(results_all) <- markets
#
# # Forecast all markets
# forecasts_all <- lapply(results_all, function(r) predictGINN(r, h=12))


# ============================================================
# Lambda interpretation (GINN)
# ============================================================
# lambda = 1.0 → Standard LSTM (ground-truth variance only, no GARCH)
# lambda = 0.5 → Balanced GINN (equal GT + GARCH weight)
# lambda = 0.0 → GINN-0 (LSTM guided entirely by GARCH)
#
# Best lambda selected automatically by LOWEST Training RMSE

# ============================================================
# Output units — important!
# ============================================================
# All variance outputs are in VARIANCE units (sigma^2), not std dev.
# To get volatility in std dev units: sqrt(forecast_var)
# Example:
fc <- predictGINN(fit, h = 12)
cat("Forecast variance    :", fc$forecast_var, "\n")
cat("Forecast volatility  :", sqrt(fc$forecast_var), "\n")
