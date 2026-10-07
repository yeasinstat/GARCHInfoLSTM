\name{plotGINN}
\alias{plotGINN}
\title{Plot GINN Volatility: Actual vs Predicted + Forecast}
\description{
Plots conditional variance (volatility) in one chart showing:
\enumerate{
  \item Actual realized variance sigma2_t on full training data (dark line).
  \item GINN in-sample predicted variance (blue line).
  \item Out-of-sample variance forecast (red solid line).
}
If actual future prices are provided, their realized variance is computed
and overlaid as a green line for direct comparison with the red forecast.
}
\usage{
plotGINN(fit, h = 12, lambda = NULL, actual = NULL,
          newdata = NULL, n_hist = NULL)
}
\arguments{
  \item{fit}{A GINN object returned by GINN().}
  \item{h}{Forecast horizon (steps ahead in variance space). Default 12.}
  \item{lambda}{Lambda to use. NULL = best lambda.}
  \item{actual}{Optional numeric vector of actual future PRICES.
    Their variance is computed and overlaid in green.}
  \item{newdata}{Optional new prices to update starting sequence.}
  \item{n_hist}{Number of historical variance points to show. NULL = all.}
}
\value{A ggplot2 object (invisibly).}
\examples{
\donttest{
prices <- cumprod(1 + rnorm(60, 0.001, 0.02)) * 100
fit    <- GINN(prices, mode = "manual", lag = 5,
                ar_lag = 1, garch_p = 1, garch_q = 1,
                garch_mean = "constant", garch_dist = "norm",
                hidden_size = 4, num_layers = 1, lr = 0.01,
                epochs = 20, patience = 5,
                lambda_list = c(0.5), verbose = FALSE)

# Plot variance forecast only
plotGINN(fit, h = 12)

# Show last 50 historical variance points only
plotGINN(fit, h = 12, n_hist = 50)

# With actual future prices overlaid (green = actual variance)
new_prices <- tail(prices, 1) * cumprod(1 + rnorm(12, 0.001, 0.02))
plotGINN(fit, h = 12, actual = new_prices)
}
}
