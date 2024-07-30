setwd("C:/Users/User/OneDrive/Desktop/TAE Comp 2024")
library(xgboost)
library(caTools)
library(dplyr)
library(caret)

# DATA EXTRACTION
df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")

df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

# FEATURE SELECTION
choice <- subset(df_train, select = c(Ch1, Ch2, Ch3, Ch4))
subset1_train <- subset(df_train, select = -c(Task, Ch1, Ch2, Ch3, Ch4, educ, gender, region, segment, ppark, night, miles, Case, No, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4, Urb, income, age))
subset1_test <- subset(df_test, select = -c(Task, Ch1, Ch2, Ch3, Ch4, educ, gender, region, segment, ppark, night, miles, Case, No, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4, Urb, income, age))

# K-FOLD CROSS VALIDATION MODEL TRAINING + PARAM SELECTION
# permanently fix subsample at 1 since we have validation
# nrounds set to 100 - from manual testing, appears that no sig changes past 100 trees if lambda significantly high enough (which provides better generalization)
# best eta = 0.07, nrounds = 100,max_depth = 5. - take ~ 1 up and 1 down for sensitivity.
# eta = 0.06 - 0.08, step = 0.1
# max_depth in range 5 - 7, step = 1
# gamma = 4 - 9, step = 1



set.seed(12)
k <- 5
folds <- createFolds(subset1_train$Choice, k = k, list = TRUE)
cv_results <- data.frame()

weight <- 0.8
final_average_val_logloss <- 10
final_average_overfitting <- 10 
final_scoring_metric <- final_average_val_logloss + (weight * final_average_overfitting) 

for (a in 5:7) {
  for(b in 5:7) {
    for(c in 6:8) {
      for(d in 0:2) {
        for(e in 0:2) {
          xgb_params <- list(
            booster = "gbtree",
            eta = (a/100),
            max_depth = b,
            gamma = c,
            alpha = d,
            lambda = e,
            subsample = 1,
            colsample_bytree = 1,
            objective = "multi:softprob",
            eval_metric = "mlogloss",
            num_class = 4
          )
          
          for (i in 1:k) {
            cat("Fold", i, "\n")
            
            train_indices <- unlist(folds[-i])
            val_indices <- folds[[i]]
            
            train_data <- subset1_train[train_indices, ]
            val_data <- subset1_train[val_indices, ]
            y_train <- as.numeric(train_data$Choice) - 1
            y_val <- as.integer(val_data$Choice) - 1
            X_train <- train_data %>% select(-Choice)
            X_val <- val_data %>% select(-Choice)
            
            xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
            xgb_val <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)
            watchlist <- list(train = xgb_train, val = xgb_val)
            
            xgb_model <- xgb.train(
              params = xgb_params,
              data = xgb_train,
              nrounds = 125,
              watchlist = watchlist,
              verbose = 1
            )
            
            eval_log <- xgb_model$evaluation_log
            last_eval_score <- eval_log[nrow(eval_log), ]
            cat("Fold", i, "Train Log Loss:", last_eval_score$train_mlogloss, "\n")
            cat("Fold", i, "Validation Log Loss:", last_eval_score$val_mlogloss, "\n")
            cv_results <- rbind(cv_results, last_eval_score)
          }
          
          # METRICS
          average_train_logloss <- mean(cv_results$train_mlogloss)
          average_val_logloss <- mean(cv_results$val_mlogloss)
          average_overfitting <- average_val_logloss - average_train_logloss
          scoring_metric <- average_val_logloss + (weight * average_overfitting) 
          
          cat("Average Training Log Loss:", average_train_logloss, "\n")
          cat("Average Validation Log Loss:", average_val_logloss, "\n")
          cat("Average Overfitting:", average_overfitting, "\n") 
          
          if (scoring_metric < final_scoring_metric) {
            eta_best <- a
            maxdepth_best <- b
            gamma_best <- c
            alpha_best <- d
            lambda_best <- e
            
            final_average_val_logloss <- average_val_logloss
            final_average_overfitting <- average_overfitting
            final_average_train_logloss <- average_train_logloss
            final_scoring_metric <- scoring_metric
          }
        }
      }
    }
  }
}

cat("Best Average Training Log Loss:", final_average_train_logloss, "\n")
cat("Best Average Validation Log Loss:", final_average_val_logloss, "\n")
cat("Best Average Overfitting:", final_average_overfitting, "\n") 
cat("With eta = ",eta_best, "\n")
cat("and maxdepth = ", maxdepth_best, "\n") 
cat("and gamma = ", gamma_best, "\n") 
cat("and alpha = ", alpha_best, "\n") 
cat("and lambda = ", lambda_best, "\n") 

# With eta =  7 
# > cat("and maxdepth = ", maxdepth_best, "\n") 
# and maxdepth =  7 
# > cat("and gamma = ", gamma_best, "\n") 
# and gamma =  8 
# > cat("and alpha = ", alpha_best, "\n") 
# and alpha =  2 
# > cat("and lambda = ", lambda_best, "\n") 
# and lambda =  2 

# # PREDICTING AND TESTING 
df_test <- read.csv("test2024.csv")
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))
subset_test <- subset(df_test,select=-c(Task,Ch1,Ch2,Ch3,Ch4,educ,gender,region,segment,ppark,night,miles,Case,CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4,Urb,income,age))

y_test <- as.integer(subset_test$Choice) -1
X_test <- subset_test %>% select(-c(Choice,No))

xgb_preds <- predict(xgb_model, as.matrix(X_test), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
colnames(xgb_preds) <- c("1","2","3","4")

xgb_preds
# write.csv(xgb_preds, "0728_submission4_xgb1.1_revisedparameters.csv")