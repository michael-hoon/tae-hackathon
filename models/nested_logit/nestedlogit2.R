# DATA SPLITTING DONE USING TASK NUMBER
library(mlogit)

safety <- read.csv("~/TAE_Code/tae-hackathon/data/train2024.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, ifelse(safety$Ch2 == 1, 2, ifelse(safety$Ch3 == 1, 3, 4)))

# ========================
#      DATA SPLITTING
# ========================
train <- subset(safety, Task<=15)
test <- subset(safety, Task>15)
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
S_train <- mlogit.data(subset(train), shape="wide", choice="Choice", varying =c(4:83), sep="", idx = list(c("No", "Case")))
# nested logit model

# first set a list of nests corresponding to the first 3 choices from the consumer being the first nest, and choice 4 being the second nest

# nests <- list(
    # Choice = c("Ch1", "Ch2","Ch3"),
    # NoChoice = c("Ch4")
# )

# here, the nests should be set via the idx$id2 choice variable entries 1 2 3 4, after transformation to wide format via dfidx
nests <- list(
    nest1 = c("1", "2", "3"),
    nest2 = c("4")
)

nestlogit <- mlogit(
    formula = Choice ~ CC + GN + NS + BU + FA + LD + BZ + FC + FP + RP + PP + KA + SC + TS + NV + MA + LB + AF + HU + Price | 0,
    data = S_train,
    nests = nests,
    un.nest.el = TRUE,
)

summary(nestlogit)
compute_logloss(nestlogit, S_train, S_test, train_choice, val_choice)
train_predictions <- predict(nestlogit, newdata=S_train)
test_predictions <- predict(nestlogit, newdata=S_test)