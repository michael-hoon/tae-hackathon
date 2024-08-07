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























