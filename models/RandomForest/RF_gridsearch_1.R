library(randomForest)
library(mlbench)
library(caret)
seed <- 1234
set.seed(seed)

safety <- read.csv("train2024_preprocessed.csv")
safety <- subset(safety, select=-c(No, Case, CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4))
trainingIndex <- createDataPartition(safety$Choice, p = 0.8, list = FALSE)
trainingSet <- safety[trainingIndex,]
testSet <- safety[-trainingIndex,]


# GRID SEARCH 
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(1:15))
metric <- "Accuracy"
rf_gridsearch <- train(factor(Choice) ~ ., data=trainingSet, method="rf", metric=metric, 
                       tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)


# RANDOM SEARCH
# Custom Objective Function 
multi_class_logloss <- function(y_true, y_prob) {
  epsilon <- 1e-15
  y_prob <- pmax(y_prob, epsilon)
  y_prob <- pmin(y_prob, 1 - epsilon)
  n <- nrow(y_true)
  logloss <- -sum(y_true * log(y_prob)) / n
  return(logloss)
}

# Not working
# custom_summary <- function(data, lev = NULL, model = NULL) {
#   # One-hot encode the true labels
#   y_true <- model.matrix(~ data$obs - 1, data = data)
#   
#   # Extract the predicted probabilities
#   y_prob <- as.matrix(data[, lev, drop = FALSE])
#   
#   # Calculate logloss for validation set
#   val_logloss <- multi_class_logloss(y_true, y_prob)
#   
#   # Train logloss (assuming you can access the training data in the same format)
#   # Here, you would need to modify this part to calculate the train logloss
#   # In this example, I'm using a placeholder value
#   train_logloss <- 0.1  # Replace this with actual calculation
#   
#   # Calculate the difference
#   logloss_diff <- val_logloss - train_logloss
#   
#   # Return the custom metric
#   out <- c(logloss_diff = logloss_diff)
#   names(out) <- "logloss_diff"
#   return(out)
# }

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
metric <- "Accuracy"
mtry <- sqrt(ncol(trainingSet))
tunegrid <- expand.grid(.mtry = mtry)

rf_default <- train(factor(Choice) ~ CC1, data = trainingSet, method = "rf", metric = metric, 
                    tuneGrid = tunegrid, trControl = control)
plot(rf_default)
















