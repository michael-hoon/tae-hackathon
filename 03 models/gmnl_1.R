setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
rm(list=ls())

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
#          GMNL
# ========================
library(gmnl)
library(mlogit)
# S_train <- dfidx(subset(train), shape="wide", choice="Choice", varying =c(4:83), sep="", idx = list(c("No", "Case")))
# S_test <- dfidx(subset(test), shape="wide", choice="Choice", varying = c(4:83), sep="", idx = list(c("No", "Case")))
car_features <- c('CC1','GN1','NS1','BU1','FA1','LD1','BZ1','FC1','FP1','RP1','PP1','KA1','SC1','TS1','NV1','MA1','LB1','AF1','HU1',
                  'CC2','GN2','NS2','BU2','FA2','LD2','BZ2','FC2','FP2','RP2','PP2','KA2','SC2','TS2','NV2','MA2','LB2','AF2','HU2',
                  'CC3','GN3','NS3','BU3','FA3','LD3','BZ3','FC3','FP3','RP3','PP3','KA3','SC3','TS3','NV3','MA3','LB3','AF3','HU3',
                  'CC4','GN4','NS4','BU4','FA4','LD4','BZ4','FC4','FP4','RP4','PP4','KA4','SC4','TS4','NV4','MA4','LB4','AF4','HU4')
S_train <- mlogit.data(subset(train), choice="Choice", shape="wide", v.names=car_features, varying=c(4:83))
S_test <- mlogit.data(subset(test), choice="Choice", shape="wide", v.names=car_features, varying=c(4:83))


mixl.hier <- gmnl(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, 
                  data=S_train, model = "mixl", ranp=c(BZ="n", FP="n", RP="n", PP="n", NV="n"))





data("TravelMode", package = "AER")
library(mlogit)
TM <- mlogit.data(TravelMode, choice = "choice", shape = "long",
                  alt.levels = c("air", "train", "bus", "car"), chid.var = "individual")
## MIXL model with observed heterogeneity
mixl.hier <- gmnl(choice ~ vcost + gcost + travel + wait | 1 | 0 | income + size - 1,
                  data = TM,
                  model = "mixl",
                  ranp = c(travel = "t", wait = "n"),
                  mvar = list(travel = c("income","size"), wait = c("income")),
                  R = 30,
                  haltons = list("primes"= c(2, 17), "drop" = rep(19, 2)))
## Get the individuals' conditional mean and their standard errors for lwage
bi.travel <- effect.gmnl(mixl.hier, par = "travel", effect = "ce")
summary(bi.travel$mean)
summary(bi.travel$sd.est)
## Get the individuals' conditional WTP of travel with respect to gcost
wtp.travel <- effect.gmnl(mixl.hier, par = "travel", effect = "wtp", wrt = "gcost")
summary(wtp.travel$mean)
summary(wtp.travel$sd.est)
## End(Not run)


