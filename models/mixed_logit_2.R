# DATA SPLITTING DONE USING TASK NUMBER
setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
library(mlogit)

safety <- read.csv("train2024.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, ifelse(safety$Ch2 == 1, 2, ifelse(safety$Ch3 == 1, 3, 4)))

# DATA SPLITTING
train <- subset(safety, Task<=12)
test <- subset(safety, Task>12)
choice <- subset(test, select = c(Ch1,Ch2,Ch3,Ch4))
train_choice <- subset(train, select = c(Ch1,Ch2,Ch3,Ch4)) 
val_choice <- subset(test, select = c(Ch1,Ch2,Ch3,Ch4)) # this is y_ij for logloss


# MODEL TRAINING
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

M5 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(BZ='n', FP='n', RP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M5)




# ========================
#     MODEL TESTING # 1
# ========================
T1 <- predict(M1, newdata=S_train) # train predictions
T3 <- predict(M3, newdata=S_train)
T4 <- predict(M4, newdata=S_train)
T5 <- predict(M5, newdata=S_train)

S_test <- dfidx(subset(test), shape="wide", choice="Choice", sep="", varying = c(4:83), idx = list(c("No", "Case")))
P1 <- predict(M1, newdata=S_test) # validation predictions
P3 <- predict(M3, newdata=S_test)
P4 <- predict(M4, newdata=S_test)
P5 <- predict(M5, newdata=S_test)

compute_logloss <- function(P, choice){
    # P: predicted choices
    # choice: actual choices
    logloss <- 0
    for (i in 1:(nrow(choice))) {
        logloss <- logloss + choice$Ch1[i]*log(P[i,1]) + choice$Ch2[i]*log(P[i,2]) + choice$Ch3[i]*log(P[i,3]) + choice$Ch4[i]*log(P[i,4])
    }
    logloss <- logloss / nrow(choice)
    logloss <- -1 * logloss
    logloss
}

M1_train_logloss <- compute_logloss(T1, train_choice)
M1_train_logloss 
M3_train_logloss <- compute_logloss(T3, train_choice)
M3_train_logloss 
M4_train_logloss <- compute_logloss(T4, train_choice)
M4_train_logloss 
M5_train_logloss <- compute_logloss(T5, train_choice)
M5_train_logloss 

M1_val_logloss <- compute_logloss(P1, val_choice)
M1_val_logloss
M3_val_logloss <- compute_logloss(P3, val_choice)
M3_val_logloss
M4_val_logloss <- compute_logloss(P4, val_choice)
M4_val_logloss
M5_val_logloss <- compute_logloss(P5, val_choice)
M5_val_logloss

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
logloss_M3 <- validation_logloss(P3)
logloss_M4 <- validation_logloss(P4)
logloss_M1
logloss_M3
logloss_M4





