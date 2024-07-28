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






