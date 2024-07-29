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
model <- randomForest(as.factor(Choice) ~ ., data = trainingSet, mtry=best.m, importance=TRUE, ntree = 2500) # Nathan might want to lower down to 1500 to try for our dataset size, 2500 too large

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
pred <- predict(model, testSet, type="prob")

# Change col names to Ch1, Ch2, Ch3, Ch4 to calculate logloss using function
colnames(pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")

# calculate logloss
logloss <- function(test_set, testpredict_df) {
    # Create one-hot encoding for each choice on-the-fly
    Ch1 <- as.integer(test_set$Choice == 1)
    Ch2 <- as.integer(test_set$Choice == 2)
    Ch3 <- as.integer(test_set$Choice == 3)
    Ch4 <- as.integer(test_set$Choice == 4)
    
    # Calculate logloss using these one-hot encoded variables
    result <- -1/nrow(test_set) * sum(Ch1 * log(testpredict_df$Ch1+.Machine$double.eps) +
                                          Ch2 * log(testpredict_df$Ch2+.Machine$double.eps) +
                                          Ch3 * log(testpredict_df$Ch3+.Machine$double.eps) +
                                          Ch4 * log(testpredict_df$Ch4+.Machine$double.eps))
    return(result)
}

# Calculate logloss
loss <- logloss(testSet, as.data.frame(pred))
loss #best result: 1.145054

# the one i calculated running this without changes (took maybe 15mins): 1.145332