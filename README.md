# GINN — GARCH-Informed Neural Network

An R package implementing the **GARCH-Informed Neural Network (GINN)** hybrid
framework for volatility (variance) forecasting.

---

## The GINN Pipeline (matches the paper exactly)

```
Step 1: y_t  →  r_t = (y_t - y_{t-1}) / y_{t-1}        [returns]
Step 2: AR(p) on r_t  →  μ̂_t                            [mean return]
Step 3: GARCH(p,q) on r_t  →  σ²̂_GARCH                 [conditional variance]
Step 4: σ²_t = (r_t - μ̂_t)²                             [ground-truth variance]
Step 5: LSTM trained on σ²_t sequences with GINN loss:
        Loss = λ × MSE(σ²_t, σ²̂_LSTM)                  [vs ground truth]
             + (1-λ) × MSE(σ²̂_GARCH, σ²̂_LSTM)          [vs GARCH  — Eq. 17]
```

**Lambda interpretation:**
- `λ = 1.0` → Standard LSTM (ground-truth variance only, no GARCH guidance)
- `λ = 0.5` → Balanced GINN (equal weight)
- `λ = 0.0` → GINN-0 (LSTM only learns from GARCH)

---

## Installation

```r
install.packages(c("torch", "rugarch", "ggplot2", "cli", "coro", "devtools"))
torch::install_torch()         # one-time ~500 MB download
devtools::install("path/to/GINN")
```

---

## Quick Start

```r
library(GINN)

# Simulated prices
prices <- cumprod(1 + rnorm(150, 0.001, 0.02)) * 100

# ── AUTO mode ─────────────────────────────────────────────────
# AR lag, GARCH config, LSTM hyperparameters all auto-selected
result <- GINN(
  data        = prices,
  mode        = "auto",
  seq_len     = 5,
  lambda_list = seq(0.1, 0.9, 0.1)
)
print(result)
plotGINN(result, h = 12)
predictGINN(result, h = 12)
plotlambda(result)
accuracy.GINN(result)   # this one stays   # full metrics table

# ── MANUAL mode ───────────────────────────────────────────────
result_m <- GINN(
  data       = prices,
  mode       = "manual",
  ar_lag     = 1,
  garch_p    = 1, garch_q = 1,
  garch_mean = "zero", garch_dist = "norm",
  seq_len    = 5, hidden_size = 32,
  num_layers = 1, lr = 0.001, dropout = 0.0,
  lambda_list = c(0.1, 0.5, 0.9)
)
print(result_m)
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `data` | required | Numeric price/yield vector |
| `mode` | `"auto"` | `"auto"` or `"manual"` |
| `ar_max_lag` | `5` | Max AR lag to search (auto) |
| `ar_lag` | `NULL` | Fixed AR lag (manual) |
| `garch_p_grid` | `1:3` | ARCH order grid (auto) |
| `garch_q_grid` | `1:3` | GARCH order grid (auto) |
| `garch_mean_grid` | `c("zero","arma")` | Mean model grid (auto) |
| `garch_dist_grid` | `c("norm","std")` | Distribution grid (auto) |
| `garch_p/q/mean/dist` | `NULL` | Fixed GARCH config (manual) |
| `seq_len` | `5` | Variance sequence length |
| `hidden_size` | `32` | LSTM hidden units (manual) |
| `num_layers` | `1` | Stacked layers (manual) |
| `lr` | `0.001` | Learning rate (manual) |
| `dropout` | `0.0` | Dropout rate (manual) |
| `epochs` | `500` | Max training epochs |
| `patience` | `30` | Early-stopping patience |
| `tune_epochs` | `50` | Epochs per grid combo (auto) |
| `batch_size` | `16` | Mini-batch size |
| `lambda_list` | `0.1…0.9` | Lambda values to test |
| `hidden_grid` | `c(8,16,24,32,40)` | Hidden grid (auto) |
| `layers_grid` | `c(1,2)` | Layers grid (auto) |
| `lr_grid` | `c(0.001,0.003,0.005)` | LR grid (auto) |
| `dropout_grid` | `c(0.0,0.1,0.2,0.3)` | Dropout grid (auto) |

---

## Output Object (`GINN` class)

```r
result$results       # list — one per lambda (RMSE, MAE, R², preds, loss_hist)
result$garch         # GARCH config, coefs, sigma²_train/test, metrics
result$ar            # AR lag, fitted values, forecast
result$returns       # computed return series r_t
result$variance      # ground-truth variance σ²_t (train/test/all)
result$best_model    # complete best model summary
result$best_hp       # best LSTM hyperparameters
result$tuning_log    # grid-search log (auto mode)
result$meta          # settings
result$data_info     # original data, split, N
```

---

## Methods

```r
print(result)                    # compact table (all lambdas)
summary(result)                  # full data frame with all metrics
plotGINN(result, h = 12)              # actual + predicted + forecast
plotGINN(result, h = 12, actual = new_prices)  # with actual future data
plotlambda(result)                    # Training RMSE bar chart
predictGINN(result, h = 12)          # variance forecast
AccuracyGINN(result, actual = new_prices)  # accuracy vs actual
plot(result, "loss")                  # training loss (unchanged)
accuracy.GINN(result)            # accuracy data frame
```

---

## Dependencies

- `torch` — LSTM (native R, no Python required)
- `rugarch` — GARCH fitting
- `ggplot2` — plots
- `cli` — progress messages
- `coro` — dataloader iteration
