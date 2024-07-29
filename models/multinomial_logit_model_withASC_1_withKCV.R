library(mlogit)
library(caret)

# DATA EXTRACTION
df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")

df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

# Create folds
set.seed(12)
k <- 5
folds <- createFolds(df_train$Choice, k = k, list = TRUE)
cv_results <- data.frame()

for (i in 1:k) {
  cat("Fold", i, "\n")
  
  train_indices <- unlist(folds[-i])
  val_indices <- folds[[i]]
  
  train <- df_train[train_indices, ]
  validate <- df_train[val_indices, ]
  
  train_probs <- train[, 110:113]
  validate_probs <- validate[, 110:113]
  
  # Format data
  S <- dfidx(train, shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))
  T_valid <- dfidx(validate, shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))
  
  # Train model (example: model with only safety features)
  M_ASC <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1 | yearind+milesind+pparkind+genderind+ageind+Urbind+incomeind, data=S)
  
  # Predicted probabilities of all safety features 
  P_vl <- predict(M_ASC, newdata=T_valid)
  
  # Mean log likelihood for this fold
  LL <- 0
  for (j in 1:nrow(P_vl)){
    for (k in 1:ncol(P_vl)){
      LL <- LL + validate_probs[j, k] * log(P_vl[j, k])
    }
  }
  LL <- -1 * (LL / nrow(P_vl))
  cv_results <- rbind(cv_results, data.frame(Fold = i, LogLoss = LL))
  cat("Fold", i, "Validation Log Loss:", LL, "\n")
}

# Average log loss across all folds
average_logloss <- mean(cv_results$LogLoss)
cat("Average Validation Log Loss:", average_logloss, "\n")

# Model Testing
S_test <- dfidx(df_test, shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))
P_test <- predict(M_onlys, newdata=S_test)

# Evaluate test log loss
test_choices <- df_test[, 110:113]
logloss <- 0
for (i in 1:nrow(test_choices)) {
  logloss <- logloss + test_choices$Ch1[i] * log(P_test[i, "1"])
  logloss <- logloss + test_choices$Ch2[i] * log(P_test[i, "2"])
  logloss <- logloss + test_choices$Ch3[i] * log(P_test[i, "3"])
  logloss <- logloss + test_choices$Ch4[i] * log(P_test[i, "4"])
}
logloss <- -1 * (logloss / nrow(test_choices))
cat("Test Log Loss:", logloss, "\n")

