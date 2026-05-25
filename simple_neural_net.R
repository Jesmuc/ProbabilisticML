# Very small neural network for 1D regression from scratch in base R.
# The goal is clarity: one hidden layer, mean squared error, backpropagation,
# and stochastic gradient descent with early stopping.

set.seed(42)  # Makes random initialization reproducible.

# Target function that generates the values we want the network to learn.
target_function <- function(x) {
  sin(x)
}

# Create training points from the target function.
x_train <- seq(-pi, pi, length.out = 60)
y_train <- target_function(x_train)

# Build the network parameters.
# input_size: number of input variables
# hidden_size: number of neurons in the hidden layer
# output_size: number of outputs
init_params <- function(input_size, hidden_size, output_size) {
  list(
    W1 = matrix(rnorm(hidden_size * input_size, sd = 0.5), nrow = hidden_size, ncol = input_size),
    b1 = matrix(0, nrow = hidden_size, ncol = 1),
    W2 = matrix(rnorm(output_size * hidden_size, sd = 0.5), nrow = output_size, ncol = hidden_size),
    b2 = matrix(0, nrow = output_size, ncol = 1)
  )
}

# Hidden-layer activation function and its derivative.
tanh_derivative <- function(z) {
  1 - tanh(z)^2
}

# Forward pass through the network.
# x: one input value stored as a 1x1 matrix
# params: list with weights and biases
# Returns both the prediction and intermediate values used by backpropagation.
forward_pass <- function(x, params) {
  z1 <- params$W1 %*% x + params$b1
  a1 <- tanh(z1)
  z2 <- params$W2 %*% a1 + params$b2
  y_hat <- z2  # Linear output for regression.

  list(x = x, z1 = z1, a1 = a1, z2 = z2, y_hat = y_hat)
}

# Mean squared error for one target value and one prediction.
mse_loss <- function(y_true, y_hat) {
  mean((y_true - y_hat)^2)
}

# Backpropagation: compute gradients of the loss with respect to all parameters.
# cache: output from forward_pass()
# y_true: true target value stored as a 1x1 matrix
backprop <- function(cache, y_true, params) {
  dL_dyhat <- 2 * (cache$y_hat - y_true)

  dW2 <- dL_dyhat %*% t(cache$a1)
  db2 <- dL_dyhat

  da1 <- t(params$W2) %*% dL_dyhat
  dz1 <- da1 * tanh_derivative(cache$z1)

  dW1 <- dz1 %*% t(cache$x)
  db1 <- dz1

  list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)
}

# Stochastic gradient descent update.
# lr: learning rate that controls the step size
sgd_step <- function(params, grads, lr) {
  params$W1 <- params$W1 - lr * grads$dW1
  params$b1 <- params$b1 - lr * grads$db1
  params$W2 <- params$W2 - lr * grads$dW2
  params$b2 <- params$b2 - lr * grads$db2
  params
}

# Predict many values by running the forward pass one sample at a time.
predict_nn <- function(x_values, params) {
  sapply(x_values, function(x) {
    x_matrix <- matrix(x, nrow = 1, ncol = 1)
    forward_pass(x_matrix, params)$y_hat[1, 1]
  })
}

# Training loop.
# x_values, y_values: vectors with training data
# hidden_size: number of hidden neurons
# lr: learning rate
# max_steps: maximum SGD iterations
# error_threshold: stop early if full-dataset MSE becomes small enough
train_nn <- function(x_values, y_values, hidden_size = 6, lr = 0.01,
                     max_steps = 20000, error_threshold = 0.001) {
  params <- init_params(input_size = 1, hidden_size = hidden_size, output_size = 1)
  history <- numeric(max_steps)

  for (step in 1:max_steps) {
    i <- sample.int(length(x_values), size = 1)

    x <- matrix(x_values[i], nrow = 1, ncol = 1)
    y <- matrix(y_values[i], nrow = 1, ncol = 1)

    cache <- forward_pass(x, params)
    grads <- backprop(cache, y, params)
    params <- sgd_step(params, grads, lr)

    predictions <- predict_nn(x_values, params)
    history[step] <- mean((y_values - predictions)^2)

    if (history[step] < error_threshold) {
      history <- history[1:step]
      cat("Stopped early at step", step, "with MSE =", history[step], "\n")
      break
    }
  }

  list(params = params, loss_history = history)
}

# Train the network on the sample points.
model <- train_nn(
  x_values = x_train,
  y_values = y_train,
  hidden_size = 8,
  lr = 0.02,
  max_steps = 30000,
  error_threshold = 0.0008
)

# Use the trained model to predict the training inputs.
y_pred <- predict_nn(x_train, model$params)

# Final error on the whole dataset.
final_mse <- mean((y_train - y_pred)^2)
cat("Final MSE:", final_mse, "\n")

# Plot the true function and the neural-network approximation.
plot(x_train, y_train, pch = 16, col = "steelblue",
     main = "Neural Network Regression in Base R",
     xlab = "x", ylab = "y")
lines(x_train, y_pred, col = "firebrick", lwd = 2)
# Use a compact legend so it does not cover too much of the plot.
legend("topright",
       inset = 0.02,
       legend = c("Target points", "NN prediction"),
       col = c("steelblue", "firebrick"),
       pch = c(16, NA),
       lty = c(NA, 1),
       lwd = c(NA, 2),
       bty = "n",
       cex = 0.9,
       pt.cex = 0.9,
       x.intersp = 0.6,
       y.intersp = 0.8,
       seg.len = 1.5)

# Plot how the loss changes during training.
plot(model$loss_history, type = "l", lwd = 2, col = "darkgreen",
     main = "Training Loss",
     xlab = "Step", ylab = "MSE")
