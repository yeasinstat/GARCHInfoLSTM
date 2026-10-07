\name{predictGINN}
\alias{predictGINN}
\title{Forecast Conditional Variance h Steps Ahead from GINN}
\description{
Forecasts future conditional variance (volatility) h steps ahead using
recursive one-step prediction on variance sequences.
Returns single-point forecasts only.
}
\usage{
predictGINN(fit, h = 1, lambda = NULL, newdata = NULL)
}
\arguments{
  \item{fit}{A GINN object returned by GINN().}

  \item{h}{Number of steps to forecast ahead. Default 1.}

  \item{lambda}{Lambda value to use. NULL = best (lowest Train RMSE).
    Use 0.0 for Standard LSTM (GT variance only, no GARCH guidance).}

  \item{newdata}{Optional numeric vector of new DATA that arrived after
    training. Their variance is computed and used to update the starting
    sequence before forecasting.}
}
\value{
Data frame with columns:
\describe{
  \item{h}{Forecast horizon step (1, 2, ..., h).}
  \item{time_idx}{Time index in return space (N_ret + 1, ..., N_ret + h).}
  \item{forecast_var}{Point forecast of conditional variance sigma2_t.}
}
}
\examples{
\donttest{
prices <- cumprod(1 + rnorm(60, 0.001, 0.02)) * 100
fit    <- GINN(prices, mode = "manual", lag = 5,
                ar_lag = 1, garch_p = 1, garch_q = 1,
                garch_mean = "constant", garch_dist = "norm",
                hidden_size = 4, num_layers = 1, lr = 0.01,
                epochs = 20, patience = 5,
                lambda_list = c(0.3, 0.7), verbose = FALSE)

# Forecast 12 steps of conditional variance
fc <- predictGINN(fit, h = 12)
print(fc)

# Volatility = sqrt(variance)
sqrt(fc$forecast_var)

# Specific lambda
fc3 <- predictGINN(fit, h = 6, lambda = 0.3)

# With new prices to update starting sequence
new_prices <- tail(prices, 1) * cumprod(1 + rnorm(5, 0.001, 0.02))
fc_new     <- predictGINN(fit, h = 12, newdata = new_prices)
}
}
