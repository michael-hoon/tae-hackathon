setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
# MULTINOMIAL LOGIT MODEL

safety <- read.csv("train2024.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, ifelse(safety$Ch2 == 1, 2, ifelse(safety$Ch3 == 1, 3, 4)))
str(safety)
summary(safety)
head(safety)

set.seed(12)
split <- sample(1:nrow(safety), 0.8*nrow(safety))
train <- safety[split,] # actual train 80%
test <- safety[-split,] # actually this is validation set 20%
choice <- subset(safety, select = c(Ch1,Ch2,Ch3,Ch4))[-split,] # this is y_ij for loglos


library(mlogit)
S_train <- dfidx(subset(safety), shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))
S_test <- dfidx(subset(test), shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))
M <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train)
summary(M)
M <- mlogit(Choice~CC+GN+BU+FA+LD+BZ+FC+RP+PP+KA+SC+TS+NV+MA+LB+HU+Price-1, data=S_train)
summary(M)


P <- predict(M, newdata=S_test)
PredictedChoice <- apply(P,1,which.max)

logloss <- 0
for (i in 1:nrow(choice)) {
  logloss <- logloss + choice$Ch1[i]*log(P[i,1]) + choice$Ch2[i]*log(P[i,2]) + choice$Ch3[i]*log(P[i,3]) + choice$Ch4[i]*log(P[i,4])
}
logloss <- logloss / nrow(choice)
logloss <- -1 * logloss
logloss

compute_logloss <- function(test_set, testpredict_df) {
    # Create one-hot encoding for each choice on-the-fly
    Ch1 <- as.integer(test_set$Choice == 1)
    Ch2 <- as.integer(test_set$Choice == 2)
    Ch3 <- as.integer(test_set$Choice == 3)
    Ch4 <- as.integer(test_set$Choice == 4)
    
    # Calculate logloss using these one-hot encoded variables
    result <- -1/nrow(test_set) * sum(Ch1 * log(testpredict_df[,1]+.Machine$double.eps) +
                                          Ch2 * log(testpredict_df[,2]+.Machine$double.eps) +
                                          Ch3 * log(testpredict_df[,3]+.Machine$double.eps) +
                                          Ch4 * log(testpredict_df[,4]+.Machine$double.eps))
    return(result)
}


compute_logloss(test, P)
