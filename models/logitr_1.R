setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
library(logitr)
library(mlogit)

safety <- read.csv("train2024.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, ifelse(safety$Ch2 == 1, 2, ifelse(safety$Ch3 == 1, 3, 4)))
# DATA SPLITTING 
set.seed(12)
split <- sample(1:nrow(safety), 0.8*nrow(safety))
train <- safety[split,] # actual train 80%
test <- safety[-split,] # actually this is validation set 20%
choice <- subset(safety, select = c(Ch1,Ch2,Ch3,Ch4))[-split,] # this is y_ij for loglos

S_train <- dfidx(subset(train), shape="long", choice="Choice", varying =c(4:83), sep="",idx = list(c("No", "Case")))
features <- c("CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP", "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU", "Price")

mnl_pref <- logitr(
    data    = S_train,
    # outcome = S_train$Choice,
    outcome = "Choice",
    obsID   = "No",
    pars    = features
)

mnl_wtp <- logitr(
    data     = yogurt,
    outcome  = "Choice",
    obsID    = "No",
    pars     = features,
    scalePar = c("Price1", "Price2", "Price3", "Price4")
)