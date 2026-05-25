library(tidyverse)
library(neuralnet)

# convert character column types into factors.
iris <- iris %>% mutate_if(is.character, as.factor)
print(summary(iris))

#set.seed(245)

# Train and Test Split
data_rows <- floor(0.80 * nrow(iris))
train_indices <- sample(c(1:nrow(iris)), data_rows)
train_data <- iris[train_indices,]
test_data <- iris[-train_indices,]

# Model training
model = neuralnet(
  Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,
  data=train_data,
  hidden=c(4,2),
  linear.output = FALSE
)

plot(model,rep = "best")

# evaluation
pred <- predict(model, test_data)
labels <- c("setosa", "versicolor", "virginca")
prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred=labels[max.col.pred.]) %>%
  select(2) %>%
  unlist()

print(table(test_data$Species, prediction_label))

# accuracy
check = as.numeric(test_data$Species) == max.col(pred)
accuracy = (sum(check)/nrow(test_data))*100
print(accuracy)