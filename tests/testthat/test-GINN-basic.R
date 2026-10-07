test_that("GARCHInfoLSTM namespace loads", {
  expect_true("GARCHInfoLSTM" %in% loadedNamespaces())
})


test_that("GINN validates insufficient input length", {
  x <- seq(100, 120, length.out = 20)

  expect_error(
    GINN(
      data = x,
      lag = 5,
      verbose = FALSE
    ),
    "requires at least 30 observations"
  )
})


test_that("GINN rejects invalid lag specification", {
  x <- seq(100, 160, length.out = 60)

  expect_error(
    GINN(
      data = x,
      lag = 0,
      verbose = FALSE
    ),
    "lag must be a positive integer"
  )
})
