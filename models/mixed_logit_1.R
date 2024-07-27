setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
# MIXED LOGIT MODEL

# DATA EXTRACTION
safety <- read.csv("train2024.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, ifelse(safety$Ch2 == 1, 2, ifelse(safety$Ch3 == 1, 3, 4)))
str(safety)
summary(safety)
head(safety)

# DATA SPLITTING # 1
set.seed(12)
split <- sample(1:nrow(safety), 0.8*nrow(safety))
train <- safety[split,] # actual train 80%
test <- safety[-split,] # actually this is validation set 20%
choice <- subset(safety, select = c(Ch1,Ch2,Ch3,Ch4))[-split,] # this is y_ij for loglos

# MODEL TRAINING # 1 
library(mlogit)
S_train <- dfidx(subset(train), shape="wide", choice="Choice", varying =c(4:83), sep="",idx = list(c("No", "Case")))

M1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train,
             rpar=c(CC='n', GN='n', NS='n', BU='n',FA='n',LD='n',BZ='n',FC='n',FP='n',RP='n',PP='n',KA='n',
                    SC='n',TS='n',NV='n',MA='n',LB='n',AF='n',HU='n',Price='n'),
             panel = TRUE, print.level=TRUE)
summary(M1)

# M2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train,
#              rpar=c(CC='ln', GN='ln', NS='ln', BU='ln',FA='ln',LD='ln',BZ='ln',FC='ln',FP='ln',RP='ln',PP='ln',KA='ln',
#                     SC='ln',TS='ln',NV='ln',MA='ln',LB='ln',AF='ln',HU='ln',Price='ln'),
#              panel = TRUE, print.level=TRUE)
# summary(M2)

M3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(CC='t', GN='t', NS='t', BU='t',FA='t',LD='t',BZ='t',FC='t',FP='t',RP='t',PP='t',KA='t',
                    SC='t',TS='t',NV='t',MA='t',LB='t',AF='t',HU='t',Price='t'), 
             panel = TRUE, print.level=TRUE)
summary(M3)

M4 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(CC='u', GN='u', NS='u', BU='u',FA='u',LD='u',BZ='u',FC='u',FP='u',RP='u',PP='u',KA='u',
                    SC='u',TS='u',NV='u',MA='u',LB='u',AF='u',HU='u',Price='u'), 
             panel = TRUE, print.level=TRUE)
summary(M4)


# MODEL TESTING # 1
S_test <- dfidx(subset(test), shape="wide", choice="Choice", sep="", varying = c(4:83), idx = list(c("No", "Case")))
P1 <- predict(M1, newdata=S_test)
# P2 <- predict(M2, newdata=S_test)
P3 <- predict(M3, newdata=S_test)
P4 <- predict(M4, newdata=S_test)
# PredictedChoice <- apply(P,1,which.max)

validation_logloss <- function(P){
    logloss <- 0
    for (i in 1:nrow(choice)) {
        logloss <- logloss + choice$Ch1[i]*log(P[i,1]) + choice$Ch2[i]*log(P[i,2]) + choice$Ch3[i]*log(P[i,3]) + choice$Ch4[i]*log(P[i,4])
    }
    logloss <- logloss / nrow(choice)
    logloss <- -1 * logloss
    logloss
}
logloss_M1 <- validation_logloss(P1)
# logloss_M2 <- validation_logloss(P2)
logloss_M3 <- validation_logloss(P3)
logloss_M4 <- validation_logloss(P4)
logloss_M1
logloss_M3
logloss_M4




# RANDOM TASTE
library(ggplot2)
features <- c("CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP", "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU", "Price")


mean_price <- random_params["price", "Estimate"]
sd_price <- random_params["price", "Std. Error"]

# Generate random samples from the normal distribution
set.seed(123)
price_samples <- rnorm(1000, mean = mean_price, sd = sd_price)

# Plot the distribution
ggplot(data.frame(price_samples), aes(x = price_samples)) +
    geom_histogram(aes(y = ..density..), bins = 30, color = "black", fill = "lightblue") +
    geom_density(color = "red") +
    labs(title = "Distribution of Random Coefficient for Price", x = "Price Coefficient", y = "Density")

# PLOTTING ERRORS
residual_1 <- test$Ch1 - P[,1]
residual_2 <- test$Ch2 - P[,2]
residual_3 <- test$Ch3 - P[,3]
residual_4 <- test$Ch4 - P[,4]
# residual_1 <- test$Ch1 - C1
# residual_2 <- test$Ch2 - C2
# residual_3 <- test$Ch3 - C3
# residual_4 <- test$Ch4 - C4

plot(residual_1)
plot(residual_2)
plot(residual_3)
plot(residual_4)





















