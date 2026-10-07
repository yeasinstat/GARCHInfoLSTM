# ============================================================
utils::globalVariables(c(
  "time_idx", "value", "Series", "forecast",
  "Lambda", "RMSE", "Best"
))
# GINN -- plotGINN + plotlambda
# ============================================================
#' @importFrom utils tail
NULL
#' Plot GINN: Actual vs Predicted Variance + Out-of-sample Forecast
#'
#' Shows in one plot:
#' 1. Actual ground-truth variance (dark line)
#' 2. In-sample predicted variance (blue line)
#' 3. Out-of-sample variance forecast (red solid, single point, no CI)
#' 4. If actual future prices provided: actual future variance (green)
#'
#' @param fit     A GINN object returned by GINN().
#' @param h       Forecast horizon. Default 12.
#' @param lambda  Lambda to use. NULL = best.
#' @param actual  Optional numeric vector of actual future PRICES.
#' @param newdata Optional new prices to update starting sequence.
#' @param n_hist  Number of historical points to show. NULL = all.
#'
#' @return ggplot2 object (invisibly).
#' @export
plotGINN <- function(fit, h = 12, lambda = NULL, actual = NULL,
                     newdata = NULL, n_hist = NULL) {

  # -- Select model ------------------------------------------
  nm <- if (is.null(lambda)) fit$meta$best_lam_nm else {
    if (lambda == 0.0) "standard" else paste0("lambda_", lambda)
  }
  lam_label <- gsub("lambda_","", nm)
  rd        <- fit$results[[nm]]

  # -- In-sample actual vs predicted variance -----------------
  tr_act  <- rd$tr_actuals
  tr_pred <- rd$tr_preds
  n_tr    <- length(tr_act)
  seq_ln  <- fit$data_info$lag
  start_i <- seq_ln + 1L
  end_i   <- start_i + n_tr - 1L

  if (!is.null(n_hist) && n_hist < n_tr) {
    keep    <- (n_tr - n_hist + 1L):n_tr
    tr_act  <- tr_act[keep]; tr_pred <- tr_pred[keep]
    start_i <- end_i - n_hist + 1L
  }
  t_idx <- start_i:end_i

  df_actual <- data.frame(time_idx=t_idx, value=tr_act,
                           Series="Actual Variance")
  df_pred   <- data.frame(time_idx=t_idx, value=tr_pred,
                           Series="Predicted (In-sample)")

  # -- Out-of-sample forecast ---------------------------------
  fc    <- predictGINN(fit, h=h, lambda=lambda, newdata=newdata)
  fc_df <- data.frame(time_idx=fc$time_idx, forecast=fc$forecast_var)

  # -- Colour palette -----------------------------------------
  cols <- c(
    "Actual Variance"       = "#2C3E50",
    "Predicted (In-sample)" = "#2980B9",
    "Forecast"              = "#E74C3C",
    "Actual (Future)"       = "#27AE60"
  )

  # -- Base plot ----------------------------------------------
  p <- ggplot2::ggplot() +

    # Actual variance -- dark
    ggplot2::geom_line(
      data=df_actual, ggplot2::aes(x=time_idx, y=value, colour=Series),
      linewidth=0.85) +
    # In-sample predicted -- blue
    ggplot2::geom_line(
      data=df_pred, ggplot2::aes(x=time_idx, y=value, colour=Series),
      linewidth=0.85) +
    # Forecast -- red solid
    ggplot2::geom_line(
      data=fc_df, ggplot2::aes(x=time_idx, y=forecast, colour="Forecast"),
      linewidth=1.1) +
    # Boundary line
    ggplot2::geom_vline(
      xintercept=fit$data_info$N_ret, linetype="dashed",
      colour="grey50", linewidth=0.6) +
    ggplot2::annotate("text",
      x=fit$data_info$N_ret, y=Inf,
      label=" Forecast", hjust=0, vjust=1.5, size=3, colour="grey40") +
    ggplot2::scale_colour_manual(values=cols, name="Series") +
    ggplot2::labs(
      x="Time Index (Return Steps)", y="Conditional Variance (sigma2_t)"
    ) +
    ggplot2::theme_minimal(base_size=11) +
    ggplot2::theme(
      plot.title      = ggplot2::element_text(face="bold"),
      legend.position = "bottom"
    )

  # -- Overlay actual future variance if provided -------------
  if (!is.null(actual)) {
    nd <- as.numeric(actual)
    all_prices <- c(fit$data_info$prices, nd)
    all_ret    <- diff(all_prices) / all_prices[-length(all_prices)]
    new_ret    <- tail(all_ret, length(nd))
    mu_new     <- mean(new_ret)
    act_var    <- (new_ret - mu_new)^2

    act_df <- data.frame(
      time_idx = fit$data_info$N_ret + seq_along(act_var),
      value    = act_var,
      Series   = "Actual (Future)"
    )
    p <- p +
      ggplot2::geom_line(
        data=act_df, ggplot2::aes(x=time_idx, y=value, colour=Series),
        linewidth=0.95) +
      ggplot2::geom_point(
        data=act_df, ggplot2::aes(x=time_idx, y=value),
        colour="#27AE60", size=2) +
      ggplot2::labs(
        caption="Dark=Actual var  |  Blue=Predicted (in-sample)  |  Red=Forecast  |  Green=Actual (future)")
  } else {
    p <- p + ggplot2::labs(
      caption="Dark=Actual variance  |  Blue=Predicted (in-sample)  |  Red=Forecast")
  }

  print(p)
  invisible(p)
}


#' Plot Training RMSE Across Lambda Values (GINN)
#'
#' @param fit   A GINN object.
#' @return ggplot2 object (invisibly).
#' @export
plotlambda <- function(fit) {

  best_nm   <- fit$meta$best_lam_nm
  all_lams  <- fit$meta$all_lams
  lam_names <- names(fit$results)

  # Use Training RMSE (consistent with best lambda selection)
  rmse_vals  <- sapply(lam_names, function(nm) fit$results[[nm]]$tr_rmse)
  lam_labels <- ifelse(all_lams==0.0, "Std\n(0.0)", paste0(all_lams))

  df <- data.frame(
    Lambda = factor(lam_labels, levels=lam_labels),
    RMSE   = rmse_vals,
    Best   = lam_names == best_nm,
    stringsAsFactors=FALSE
  )

  best_rmse <- min(rmse_vals)
  best_lbl  <- gsub("lambda_","", best_nm)

  p <- ggplot2::ggplot(df, ggplot2::aes(x=Lambda, y=RMSE, fill=Best)) +
    ggplot2::geom_bar(stat="identity", alpha=0.85, width=0.7) +
    ggplot2::scale_fill_manual(
      values=c("FALSE"="#3498DB","TRUE"="#FFD700"), guide="none") +
    ggplot2::geom_hline(
      yintercept=best_rmse, linetype="dashed",
      colour="#E74C3C", linewidth=0.7) +
    ggplot2::geom_text(
      ggplot2::aes(label=format(round(RMSE,8), scientific=TRUE)),
      vjust=-0.4, size=2.6, colour="grey30") +
    ggplot2::coord_cartesian(
      ylim = c(min(rmse_vals) * 0.985, max(rmse_vals) * 1.03)
    ) +
    ggplot2::labs(
      x="Lambda", y="Training RMSE (Variance Scale)",
      caption="Red dashed = minimum RMSE  |  Gold = best lambda"
    ) +
    ggplot2::theme_minimal(base_size=11) +
    ggplot2::theme(plot.title=ggplot2::element_text(face="bold"))

  print(p)
  invisible(p)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
