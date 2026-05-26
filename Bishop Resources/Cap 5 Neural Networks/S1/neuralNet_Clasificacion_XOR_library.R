# Install required package if not already installed
if (!require(neuralnet)) {
  install.packages("neuralnet")
  library(neuralnet)
}

# -----------------------------
# 1. Prepare the dataset (XOR)
# -----------------------------
# XOR truth table
xor_data <- data.frame(
  input1 = c(0, 0, 1, 1),
  input2 = c(0, 1, 0, 1),
  output = c(0, 1, 1, 0)
)

# -----------------------------
# 2. Train the neural network
# -----------------------------
# hidden = c(2) means 1 hidden layer with 2 neurons
# linear.output = FALSE for classification
nn_model <- neuralnet(
  output ~ input1 + input2,
  data = xor_data,
  hidden = c(2),
  act.fct = "logistic",
  linear.output = FALSE,
  stepmax = 1e8, # increase if convergence issues
  rep = 100,
  algorithm = "rprop+"
)

# -----------------------------
# 3. Visualize the network
# -----------------------------
plot(nn_model,rep = "best")

# -----------------------------
# 4. Make predictions
# -----------------------------
predictions <- compute(nn_model, xor_data[, c("input1", "input2")])
predicted_values <- ifelse(predictions$net.result > 0.5, 1, 0)

# -----------------------------
# 5. Display results
# -----------------------------
results <- data.frame(
  input1 = xor_data$input1,
  input2 = xor_data$input2,
  expected = xor_data$output,
  predicted = predicted_values
)

print(results)
