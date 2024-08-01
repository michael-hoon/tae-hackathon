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
      
    
      # K-FOLD CROSS VALIDATION MODEL TRAINING
      
      ##SEED SELECTION
      for(p in 12:12){
        set.seed(p)
        
        k <- 5
        folds <- createFolds(subset1_train$Choice, k = k, list = TRUE)
        cv_results <- data.frame()
        
        xgb_params <- list(
          booster = "gbtree",
          eta = 0.08,
          max_depth = 7,
          gamma = 4,
          subsample = 1,
          colsample_bytree = 1,
          lambda = 3,
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
            nrounds = 150,
            watchlist = watchlist,
            verbose = 1,
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
        scoring_metric <- (average_val_logloss + 0.8 * average_overfitting)
        
        cat("Average Training Log Loss:", average_train_logloss, "\n")
        cat("Average Validation Log Loss:", average_val_logloss, "\n")
        cat("Average Overfitting:", average_overfitting, "\n")  
        cat("Scoring Metric:", scoring_metric, "\n") 
      }
    
      
      # # PREDICTING AND TESTING 
      df_test <- read.csv("test2024.csv")
      df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))
      subset_test <- subset(df_test,select=-c(Task,Ch1,Ch2,Ch3,Ch4,educ ,gender,region,segment,ppark,night,miles,Case,CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4,Urb,income,age))

      y_test <- as.integer(subset_test$Choice) -1
      X_test <- subset_test %>% select(-c(Choice,No))

      xgb_preds <- predict(xgb_model, as.matrix(X_test), reshape = TRUE)
      xgb_preds <- as.data.frame(xgb_preds)
      colnames(xgb_preds) <- c("1","2","3","4")
      #
      xgb_preds
      write.csv(xgb_preds, "0108_submission11_xgb1.1.5_endofhyppara.csv")
      
      
      