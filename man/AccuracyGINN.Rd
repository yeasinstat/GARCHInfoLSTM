\name{AccuracyGINN}
\alias{AccuracyGINN}
\title{Out-of-Sample Accuracy for GINN Volatility Forecasts}
\description{
Compares GINN conditional variance forecasts against realized variance
computed from actual data that arrived after training.
Returns RMSE, MAE, R-squared and SMAPE on variance units.
}
\usage{
AccuracyGINN(fit, actual, lambda = NULL, verbose = TRUE)
}
\arguments{
  \item{fit}{A GINN object returned by GINN().}
  \item{actual}{Numeric vector of actual DATA observed after training.
    Their returns and realized variance are computed internally.
    Needs at least 2 values.}
  \item{lambda}{Lambda value to use. NULL = best lambda.}
  \item{verbose}{Logical. If TRUE (default), prints a formatted accuracy
    report to the console. Set to FALSE to suppress console output and
    only receive the returned data frame.}
}
\value{
Data frame with columns: Lambda, h, RMSE, MAE, R2, SMAPE (all in variance units).
Also prints a formatted accuracy table to the console.
The data frame has a "comparison" attribute with step-by-step
Actual_Var vs Forecast_Var vs Error columns.
}
\examples{
\donttest{
prices <- cumprod(1 + rnorm(60, 0.001, 0.02)) * 100
fit    <- GINN(prices, mode = "manual", lag = 5,
                ar_lag = 1, garch_p = 1, garch_q = 1,
                garch_mean = "constant", garch_dist = "norm",
                hidden_size = 4, num_layers = 1, lr = 0.01,
                epochs = 20, patience = 5,
                lambda_list = c(0.5), verbose = FALSE)

# Simulate actual future prices
new_prices <- tail(prices, 1) * cumprod(1 + rnorm(12, 0.001, 0.02))

acc <- AccuracyGINN(fit, actual = new_prices)
print(acc)

# Step-by-step comparison
attr(acc, "comparison")
}
}
