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
#      MODEL TRAINING
# ========================
S_train <- dfidx(subset(train), shape="wide", choice="Choice", varying =c(4:83), sep="", idx = list(c("No", "Case")))
S_test <- dfidx(subset(test), shape="wide", choice="Choice", varying = c(4:83), sep="", idx = list(c("No", "Case")))


# ========================
#        LOGNORMAL
# ========================
# without ASC
M4.1.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(BZ='ln', FP='ln', RP='ln', PP='n', NV='ln'), panel = TRUE, print.level=TRUE)
summary(M4.1.1)
compute_logloss(M4.1.1, S_train, S_test, train_choice, val_choice)

# with ASC [DOES NOT WORK]
# M4.1.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
#                rpar=c(BZ='ln', FP='ln', RP='ln', PP='n', NV='ln'), panel = TRUE, print.level=TRUE)
# summary(M4.1.2)
# compute_logloss(M4.1.2, S_train, S_test, train_choice, val_choice)

# ========================
#       TRIANGULAR
# ========================
# without ASC
M4.2.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
               rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.2.1)
compute_logloss(M4.2.1, S_train, S_test, train_choice, val_choice)

# with ASC
M4.2.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
               rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.2.2)
compute_logloss(M4.2.2, S_train, S_test, train_choice, val_choice)

# without CC, with ASC
M4.2.3 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train, 
                 rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.2.3)
compute_logloss(M4.2.3, S_train, S_test, train_choice, val_choice)

# ========================
#         UNIFORM 
# ========================
# without ASC
M4.3.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train,
               rpar=c(BZ='u', FP='u', RP='u', PP='u', NV='u'), panel = TRUE, print.level=TRUE)
summary(M4.3.1)
compute_logloss(M4.3.1, S_train, S_test, train_choice, val_choice)

# with ASC
M4.3.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train,
                 rpar=c(BZ='u', FP='u', RP='u', PP='u', NV='u'), panel = TRUE, print.level=TRUE)
summary(M4.3.2)
compute_logloss(M4.3.2, S_train, S_test, train_choice, val_choice)

# without CC, with ASC
M4.3.3 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train,
                 rpar=c(BZ='u', FP='u', RP='u', PP='u', NV='u'), panel = TRUE, print.level=TRUE)
summary(M4.3.3)
compute_logloss(M4.3.3, S_train, S_test, train_choice, val_choice)

# ========================
#         NORMAL
# ========================
# without ASC
M4.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train,
                 rpar=c(BZ='n', FP='n', RP='n', PP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M4.4.1)
compute_logloss(M4.4.1, S_train, S_test, train_choice, val_choice)

# with ASC
M4.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train,
                 rpar=c(BZ='n', FP='n', RP='n', PP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M4.4.2)
compute_logloss(M4.4.2, S_train, S_test, train_choice, val_choice)

# without CC, with ASC
M4.4.3 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price, data=S_train,
                 rpar=c(BZ='n', FP='n', RP='n', PP='n', NV='n'), panel = TRUE, print.level=TRUE)
summary(M4.4.3)
compute_logloss(M4.4.3, S_train, S_test, train_choice, val_choice)

# ========================
#  EXPLORING DEMOGRAPHICS
# ========================
# without ASC, age+incomea
M4.5.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1 | agea+incomea, data=S_train, 
               rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.1)
compute_logloss(M4.5.1, S_train, S_test, train_choice, val_choice)

# with ASC, age+incomea
M4.5.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price | agea+incomea, data=S_train, 
               rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.2)
compute_logloss(M4.5.2, S_train, S_test, train_choice, val_choice)

# with ASC, all demographic variables
M4.5.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price 
                 |agea+incomea+segmentind+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
                 data=S_train, rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.3)
compute_logloss(M4.5.3, S_train, S_test, train_choice, val_choice)

# without ASC, all demographic variables
M4.5.3.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
                 |agea+incomea+segmentind+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
                 data=S_train, rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.3.1)
compute_logloss(M4.5.3.1, S_train, S_test, train_choice, val_choice)

# without ASC, all except segment 
M4.5.4 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
                   |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
                   data=S_train, rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.4)
compute_logloss(M4.5.4, S_train, S_test, train_choice, val_choice)

# without ASC, all except segment 
M4.5.4.1 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
                 |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
                 data=S_train, rpar=c(BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.4.1)
compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# ============================================
# FEATURE SELECTION FOR RANDOM PARAMETERS
# ROUND 1
# ============================================
M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
                   |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
                   data=S_train, rpar=c(NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.4.1)
compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(GN='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(BU='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(FA='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(LD='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(FC='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(KA='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(SC='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(TS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(MA='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind, 
#                    data=S_train, rpar=c(LB='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.1)
# compute_logloss(M4.5.4.1, S_train, S_test, train_choice, val_choice)

# ============================================
# FEATURE SELECTION FOR RANDOM PARAMETERS
# ROUND 2
# ============================================
M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
                   |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
                   data=S_train, rpar=c(CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.4.2)
compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)

# M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(GN='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.2)
# compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)

# M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(BU='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.2)
# compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(FA='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.2)
# compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)

# M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(LD='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.2)
# compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(FC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.2)
# compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(KA='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.2)
# compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(SC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.2)
# compute_logloss(M4.5.4.2, S_train, S_test, train_choice, val_choice)


# ============================================
# FEATURE SELECTION FOR RANDOM PARAMETERS
# ROUND 3
# ============================================
M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
                   |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
                   data=S_train, rpar=c(LB='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
summary(M4.5.4.3)
compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(GN='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(BU='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(FA='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(LD='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(FC='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(KA='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(SC='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(TS='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)
# 
# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(MA='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(AF='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)

# M4.5.4.3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1
#                    |agea+incomea+yearind+milesa+nighta+pparkind+genderind+educind+regionind+Urbind,
#                    data=S_train, rpar=c(HU='t', CC='t', NS='t',BZ='t', FP='t', RP='t', PP='t', NV='t'), panel = TRUE, print.level=TRUE)
# summary(M4.5.4.3)
# compute_logloss(M4.5.4.3, S_train, S_test, train_choice, val_choice)


# ============================================
# PREDICTIONS
# ============================================
df_test <- read.csv("test2024.csv")
df_test$Choice <- 0
S_test <- dfidx(df_test, shape="wide", choice="Choice", varying = c(4:83), sep="", idx = list(c("No", "Case")))
test_predictions <- predict(M4.5.4.3, newdata=S_test)

# write.csv(test_predictions, "submission9.csv")







