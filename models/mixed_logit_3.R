# DATA SPLITTING DONE USING TASK NUMBER
setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
library(mlogit)

safety <- read.csv("train2024.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, ifelse(safety$Ch2 == 1, 2, ifelse(safety$Ch3 == 1, 3, 4)))

# ========================
#      DATA SPLITTING
# ========================
train <- subset(safety, Task<=12)
test <- subset(safety, Task>12)
choice <- subset(test, select = c(Ch1,Ch2,Ch3,Ch4))
train_choice <- subset(train, select = c(Ch1,Ch2,Ch3,Ch4)) 
val_choice <- subset(test, select = c(Ch1,Ch2,Ch3,Ch4)) # this is y_ij for logloss



# FUNCTION TO COMPUTE BOTH TRAINING AND VALIDATION LOGLOSS
compute_logloss <- function(model, X_train, X_test, choice_train, choice_test, returnPreds=FALSE){
  # model         : your model 
  # X_train       : train data to create train predictions 
  # X_test        : test data to create test predictions 
  # choice_train  : actual predictions for the train set 
  # choice_test   : actual predictions for the test set 
  
  train_predictions <- predict(model, newdata=X_train)
  test_predictions <- predict(model, newdata=X_test)
  
  train_logloss <- 0
  for (i in 1:(nrow(train_predictions))) {
    train_logloss <- train_logloss + (choice_train$Ch1[i]*log(train_predictions[i,1])  
                                      + choice_train$Ch2[i]*log(train_predictions[i,2]) 
                                      + choice_train$Ch3[i]*log(train_predictions[i,3]) 
                                      + choice_train$Ch4[i]*log(train_predictions[i,4]))
  }
  train_logloss <- -1 * (train_logloss / nrow(train_predictions))
  
  test_logloss <- 0
  for (i in 1:(nrow(test_predictions))) {
    test_logloss <- test_logloss + (choice_test$Ch1[i]*log(test_predictions[i,1]) 
                                    + choice_test$Ch2[i]*log(test_predictions[i,2]) 
                                    + choice_test$Ch3[i]*log(test_predictions[i,3]) 
                                    + choice_test$Ch4[i]*log(test_predictions[i,4]))
  }
  test_logloss <- -1 * (test_logloss / nrow(test_predictions))
  
  cat("Train Logloss", train_logloss, "\n")
  cat("Validation Logloss", test_logloss)

  if (returnPreds == TRUE){
    return(list(train_predictions = train_predictions, test_predictions = test_predictions))
  }
}

# ========================
#      MODEL TRAINING
# ========================
S_train <- dfidx(subset(train), shape="wide", choice="Choice", varying =c(4:83), sep="", idx = list(c("No", "Case")))
S_test <- dfidx(subset(test), shape="wide", choice="Choice", varying = c(4:83), sep="", idx = list(c("No", "Case")))

# Mark 3.1 
M1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(BZ='n', FP='n', RP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M1)
compute_logloss(M1, S_train, S_test, train_choice, val_choice)


# Mark 3.2
M2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(BZ='n', RP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M2)
compute_logloss(M2, S_train, S_test, train_choice, val_choice)


# Mark 3.3
M3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
             rpar=c(BZ='n', RP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M3)
compute_logloss(M3, S_train, S_test, train_choice, val_choice)


# Mark 3.4
M4 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
             rpar=c(BZ='n', FP='n', RP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M4)
compute_logloss(M4, S_train, S_test, train_choice, val_choice)


# Mark 3.5
M5 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
             rpar=c(BZ='n', FP='n', RP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M5)
compute_logloss(M5, S_train, S_test, train_choice, val_choice)


# Mark 3.6
M6 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
             rpar=c(BZ='t', FP='t', RP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M6)
compute_logloss(M6, S_train, S_test, train_choice, val_choice)

# Mark 3.7
M7 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
             rpar=c(BZ='u', FP='u', RP='u', NV='u'), panel = TRUE, print.level=TRUE)
summary(M7)
compute_logloss(M7, S_train, S_test, train_choice, val_choice)

# Mark 3.8
M8 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
             rpar=c(BZ='cn', FP='cn', RP='cn', NV='cn'), panel = TRUE, print.level=TRUE)
summary(M8)
compute_logloss(M8, S_train, S_test, train_choice, val_choice)

# Mark 3.9
M9 <- mlogit(Choice~GN+NS+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
             rpar=c(BZ='n', FP='n', RP='cn', NV='cn'), panel = TRUE, print.level=TRUE)
summary(M9)
compute_logloss(M9, S_train, S_test, train_choice, val_choice)










