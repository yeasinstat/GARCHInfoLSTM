# ============================================================
utils::globalVariables(c("Epoch", "Loss", "Type", "Lambda", "self"))
# GINN -- S3 Methods: print, summary, plot, predict
# ============================================================

#' @export
print.GINN <- function(x, ...) {
  g <- x$garch$config
  cat("\n=============================================\n")
  cat("  GINN \u2014 GARCH-Informed Neural Network\n")
  cat("  Training on FULL data | Best by Train RMSE\n")
  cat("=============================================\n")
  cat(sprintf("  Mode       : %s\n", toupper(x$meta$mode)))
  cat(sprintf("  N (prices) : %d  |  N (returns): %d\n",
              x$data_info$N, x$data_info$N_ret))
  cat(sprintf("  AR lag     : %d\n", x$ar$lag))
  cat(sprintf("  GARCH      : GARCH(%d,%d) mean=%s dist=%s\n",
              g$p, g$q, g$mean, g$dist))
  cat(sprintf("  Best HP    : hidden=%d  layers=%d  lr=%.4f  dropout=%.1f\n",
              x$best_hp$hidden, x$best_hp$layers,
              x$best_hp$lr, x$best_hp$dropout))
  cat(sprintf("  Best lambda: %s  (by Train RMSE)\n\n",
              gsub("lambda_","",x$meta$best_lam_nm)))

  cat(sprintf("  %-10s | %14s | %12s | %10s\n",
              "Lambda", "RMSE", "MAE", "SMAPE%"))
  cat(paste(rep("-", 55), collapse=""), "\n")
  for (nm in names(x$results)) {
    rd  <- x$results[[nm]]
    lam <- rd$lambda
    lbl <- if (lam==0.0) "Std(0.0)  " else sprintf("%-10.1f", lam)
    cat(sprintf("  %s| %14.8f | %12.8f | %10.4f\n",
                lbl, rd$tr_rmse, rd$tr_mae, rd$tr_smape %||% 0))
  }
  cat("=============================================\n")
  cat("  Use predictGINN(fit, h=12) to forecast\n")
  cat("  Use fit$best_model for full details\n")
  cat("=============================================\n\n")
  invisible(x)
}


#' @export
summary.GINN <- function(object, ...) {
  cat("\n=== GINN Summary (Full-Data Training) ===\n")
  rows <- lapply(names(object$results), function(nm) {
    rd <- object$results[[nm]]
    data.frame(
      Lambda = rd$lambda,
      RMSE   = rd$tr_rmse,
      MAE    = rd$tr_mae,
      SMAPE  = round(rd$tr_smape %||% 0, 4),
      Time_s = round(rd$time, 2),
      stringsAsFactors=FALSE
    )
  })
  df <- do.call(rbind, rows)
  print(df, row.names=FALSE)
  cat(sprintf("\nBest lambda: %s  (Train RMSE = %.10f)\n",
              gsub("lambda_","",object$meta$best_lam_nm),
              min(sapply(object$results, function(r) r$tr_rmse))))
  cat("Use predictGINN(fit, h=12) for out-of-sample variance forecast\n\n")
  invisible(df)
}


#' @export
plot.GINN <- function(x, which=c("loss","lambda"), ...) {
  which <- match.arg(which)
  if (which=="lambda") return(plotlambda(x))

  df_loss <- do.call(rbind, lapply(names(x$results), function(nm) {
    rd  <- x$results[[nm]]
    h   <- rd$loss_hist
    lbl <- if (rd$lambda==0.0) "Standard" else paste0("lambda=",rd$lambda)
    ep  <- seq_along(h$total)
    rbind(
      data.frame(Epoch=ep, Loss=h$total,  Type="Total",  Lambda=lbl),
      data.frame(Epoch=ep, Loss=h$gt,     Type="GT Term", Lambda=lbl),
      data.frame(Epoch=ep, Loss=h$garch,  Type="GARCH Term",Lambda=lbl)
    )
  }))

  p <- ggplot2::ggplot(df_loss, ggplot2::aes(x=Epoch, y=Loss, colour=Type)) +
    ggplot2::geom_line(linewidth=0.5, alpha=0.8) +
    ggplot2::facet_wrap(ggplot2::vars(Lambda), scales="free_y") +
    ggplot2::labs(title="GINN \u2014 Training Loss History",
                  x="Epoch", y="Loss") +
    ggplot2::theme_minimal(base_size=9)
  print(p)
  invisible(p)
}


#' @export
predict.GINN <- function(object, h=12, lambda=NULL, ...) {
  predictGINN(object, h=h, lambda=lambda)
}


#' @keywords internal
accuracy.GINN <- function(x) {
  rows <- lapply(names(x$results), function(nm) {
    rd <- x$results[[nm]]
    data.frame(Model="GINN_LSTM", Lambda=rd$lambda,
               Train_RMSE=rd$tr_rmse, Train_MAE=rd$tr_mae,
               Train_R2=rd$tr_r2 %||% NA, Val_RMSE=rd$val_rmse)
  })
  do.call(rbind, rows)
}

`%||%` <- function(a, b) if (!is.null(a) && !is.na(a)) a else b
