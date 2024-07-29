rm(list=ls())
# Libraries
library(randomForest)
library(caret)
library(ggplot2)
library(tidyverse)

# read data
safety <- read.csv("train2024_preprocessed.csv")

# remove unnecessary columns
safety <- subset(safety, select=-c(No, Case, CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4))

# set seed 
seed <- 123

# Use caret library to create partition with 80:20 split (we might want to try KFCV here too @Nathan)
set.seed(seed)
trainingIndex <- createDataPartition(safety$Choice, p = 0.8, list = FALSE)

# Get train and test set
trainingSet <- safety[trainingIndex,]
testSet <- safety[-trainingIndex,]

# Use tuneRF function to find optimal 'mtry' parameter
mtry <- tuneRF(trainingSet[1:ncol(trainingSet)-1], as.factor(trainingSet$Choice),
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=FALSE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1] #13

# fitting model
set.seed(seed)
model <- randomForest(as.factor(Choice) ~ ., data = trainingSet, mtry=best.m, importance=TRUE, ntree = 1500) 
# Nathan might want to lower down to 1500 to try for our dataset size, 2500 too large

# Results
# importance(model)
# model
varImpPlot(model)

# variable importance
imp = as.data.frame(importance(model))
imp = cbind(vars=rownames(imp), imp)
imp = imp[order(imp$MeanDecreaseAccuracy),]
imp$vars = factor(imp$vars, levels=unique(imp$vars))

imp %>% 
    pivot_longer(cols=matches("Mean")) %>% 
    ggplot(aes(value, vars)) +
    geom_col() +
    geom_text(aes(label=round(value), x=0.5*value), size=3, colour="white") +
    facet_grid(. ~ name, scales="free_x") +
    scale_x_continuous(expand=expansion(c(0,0.04))) +
    theme_bw() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          axis.title=element_blank())

# plot error by tree size, although larger tree sizes for RF might overfit, so prev value of 2500 need to tune @Nathan
plot(x = 1:nrow(model$err.rate), y = model$err.rate[,1], type='l', ylab = "error", xlab = "trees") 

# Predict
test_pred <- predict(model, testSet, type="prob")
train_pred <- predict(model, trainingSet, type="prob")

# Change col names to Ch1, Ch2, Ch3, Ch4 to calculate logloss using function
colnames(test_pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")
colnames(train_pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")

# calculate logloss
logloss <- function(actual, predictions) {
    # Create one-hot encoding for each choice on-the-fly
    Ch1 <- as.integer(actual$Choice == 1)
    Ch2 <- as.integer(actual$Choice == 2)
    Ch3 <- as.integer(actual$Choice == 3)
    Ch4 <- as.integer(actual$Choice == 4)
    
    # Calculate logloss using these one-hot encoded variables
    result <- -1/nrow(actual) * sum(Ch1 * log(predictions$Ch1+.Machine$double.eps) +
                                    Ch2 * log(predictions$Ch2+.Machine$double.eps) +
                                    Ch3 * log(predictions$Ch3+.Machine$double.eps) +
                                    Ch4 * log(predictions$Ch4+.Machine$double.eps))
    return(result)
}

# Calculate logloss
train_loss <- logloss(trainingSet, as.data.frame(train_pred))
train_loss
test_loss <- logloss(testSet, as.data.frame(test_pred))
test_loss 

