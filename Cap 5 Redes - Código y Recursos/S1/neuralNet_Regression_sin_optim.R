# ------------------------------------------------------------
# Minimal neural network
#   1) fit a single function with a neural network
#   2) visualize the optimization history
#   3) No libraries needed, just optim()
# ------------------------------------------------------------

set.seed(123)

# -------------------------
# 1) Training data
# -------------------------
# Input points in [-1, 1]
x <- seq(-1, 1, length.out = 80)

# Target function to approximate
# It could be another function, x^2, abs(x), etc.
y <- sin(pi * x)

# -------------------------
# 2) Network architecture
# -------------------------
# 2 adaptive layers:
# 1 input  -> H hidden tanh units -> 1 linear output
H <- 8

# Total parameters:
# W1: H weights from input to hidden
# b1: H biases for hidden layer
# W2: H weights from hidden to output
# b2: 1 bias for output
n_params <- H + H + H + 1

# -------------------------
# 3) Helper to unpack parameters
# -------------------------
unpack_par <- function(par, H) {
  i <- 1
  W1 <- par[i:(i + H - 1)] ; i <- i + H
  b1 <- par[i:(i + H - 1)] ; i <- i + H
  W2 <- par[i:(i + H - 1)] ; i <- i + H
  b2 <- par[i]
  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
}

# -------------------------
# 4) Forward pass (or forward propagation)
# -------------------------
# For each x:
#   hidden_j = tanh(W1_j * x + b1_j)
#   y_hat    = sum_j W2_j * hidden_j + b2
nn_predict <- function(par, x, H) {
  p <- unpack_par(par, H)

  # outer(x, p$W1) builds all linear combinations x * W1_j
  A1 <- outer(x, p$W1) + matrix(p$b1, nrow = length(x), ncol = H, byrow = TRUE)
  Z1 <- tanh(A1)
  y_hat <- Z1 %*% p$W2 + p$b2

  as.numeric(y_hat)
}

# -------------------------
# 5) Loss function (for Regression)
# -------------------------
# Mean squared error between prediction and target
mse_loss <- function(par, x, y, H) {
  y_hat <- nn_predict(par, x, H)
  mean((y_hat - y)^2)
}

# -------------------------
# 6) Optimization
# -------------------------
# Small random initialization
par0 <- rnorm(n_params, sd = 0.5)

# We will store the loss every time optim() evaluates it
loss_history <- numeric(0)

objective_with_history <- function(par, x, y, H) {
  loss <- mse_loss(par, x, y, H)
  loss_history <<- c(loss_history, loss)
  loss
}

# Train the network
fit <- optim(
  par = par0,
  fn = objective_with_history,
  x = x,
  y = y,
  H = H,
  method = "BFGS",
  control = list(maxit = 300, reltol = 1e-10)
)

# Final predictions with trained parameters
y_hat <- nn_predict(fit$par, x, H)

# -------------------------
# 7) Visualization 1: fitted function
# -------------------------
plot(
  x, y,
  pch = 16,
  main = "Minimal neural network fit",
  xlab = "x",
  ylab = "y",
  cex = 0.7
)
lines(x, y_hat, lwd = 2)
legend(
  "topleft",
  legend = c("Target function", "Network fit"),
  pch = c(16, NA),
  lty = c(NA, 1),
  lwd = c(NA, 2),
  bty = "n"
)

# -------------------------
# 8) Visualization 2: optimization history
# -------------------------
plot(
  loss_history,
  type = "l",
  lwd = 2,
  main = "Optimization history",
  xlab = "Function evaluation",
  ylab = "MSE loss"
)

# Optional: print final loss
cat("Final loss:", fit$value, "\n")
