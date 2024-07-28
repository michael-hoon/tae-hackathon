setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
library(dplyr)
library(xgboost)
library(splitTools)
library(caret)

# DATA EXTRACTION & FEATURE ENGINEERING
train_raw <- read.csv("train2024.csv")
train_raw$Choice <- ifelse(train_raw$Ch1 == 1, 1,
                        ifelse(train_raw$Ch2 == 1, 2,
                               ifelse(train_raw$Ch3 == 1, 3,
                                      ifelse(train_raw$Ch4 == 1, 4, NA))))
df <- subset(train_raw, select = -c(segmentind,year,miles,milesind,night,nightind,
                                 pparkind,genderind,age,ageind,educind,regionind,
                                 Urbind,income,incomeind,Ch1,Ch2,Ch3,Ch4))

car_features <- c('CC1','GN1','NS1','BU1','FA1','LD1','BZ1','FC1','FP1','RP1','PP1','KA1','SC1','TS1','NV1','MA1','LB1','AF1','HU1',
  'CC2','GN2','NS2','BU2','FA2','LD2','BZ2','FC2','FP2','RP2','PP2','KA2','SC2','TS2','NV2','MA2','LB2','AF2','HU2',
  'CC3','GN3','NS3','BU3','FA3','LD3','BZ3','FC3','FP3','RP3','PP3','KA3','SC3','TS3','NV3','MA3','LB3','AF3','HU3',
  'CC4','GN4','NS4','BU4','FA4','LD4','BZ4','FC4','FP4','RP4','PP4','KA4','SC4','TS4','NV4','MA4','LB4','AF4','HU4')

for (car_feature in car_features){
  df[,car_feature] <- factor(df[,car_feature])
} 

df$CC1 <- factor(df$CC1)

dummy <- dummyVars(" ~ .", data=df)
newdata <- data.frame(predict(dummy, newdata = df)) 
# Remove 1 indicator variable for each category to reduce model variability
train <- subset(newdata, select = -c(segmentSmall.Car,pparkNever, genderFemale,
                                     educTrade.Vocational.School, regionW, UrbSuburban))
# FOR MAKING PREDICTIONS
test_raw <- read.csv("test2024.csv")
test_raw$Choice <- 0 
df <- subset(test_raw, select = -c(segmentind,year,miles,milesind,night,nightind,
                                   pparkind,genderind,age,ageind,educind,regionind,
                                   Urbind,income,incomeind,Ch1,Ch2,Ch3,Ch4))
dummy <- dummyVars(" ~ .", data=df)
newdata <- data.frame(predict(dummy, newdata = df)) 
test <- subset(newdata, select = -c(pparkNever, genderFemale, educTrade.Vocational.School, 
                                    regionW, UrbSuburban))

# DATA PREPARATION
train$Choice <- as.factor(train$Choice)
choices <- train$Choice
label <- as.integer(train$Choice)-1
train$Choice <- NULL
train <- subset(train, select = -c(Case, No))
train <- subset(train, select = -c(CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,
                                   KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4))

set.seed(123)
split <- createDataPartition(pure$Choice, p = 0.8, list = FALSE)
train.data = as.matrix(train[split, ])
train.label = label[split]
val.data = as.matrix(train[-split, ])
val.label = label[-split]

# tempno <- pure$No[-split]
# tempchoice <- pure$Choice[-split]
# tempfullno <- pure$No
# tempfullchoice <- pure$Choice
# tempfulltrain <- totrain

xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)
xgb.fulltrain = xgb.DMatrix(data=as.matrix(totrain),label=label)


# GRID SEARCH FOR HYPERPARAMETER TUNING
grid <- expand.grid(nrounds = c(50, 100, 150), ## times the algo add a new decision tree to the ensemble model, higher more complex
                    max_depth = c(5),
                    eta = c(0.01, 0.1, 0.3), ## steps taken when updating the weights, higher faster
                    gamma = c(0, 2, 4), ## minimum loss reduction required to further partition, higher more simple
                    colsample_bytree = c(0.3, 0.5, 0.7, 0.9, 1), ## what fraction of features are used to train each tree, higher means all
                    min_child_weight = c(1, 3, 5), ## decides if new child will be better, higher more simple
                    subsample = c(0.6, 0.7, 0.8)) ## what fraction of train data are used to train each tree

ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)


# MODEL TRAINING
xgb.fit <- train(x = train.data,
                 y = train.label,
                 method = "xgbTree",
                 trControl = ctrl,
                 tuneGrid = grid,
                 verbose = FALSE)







