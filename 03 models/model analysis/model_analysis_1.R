setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models\\model_analysis")
library(xgboost)
library(caTools)
library(dplyr)
library(caret)

# =======================
# DATA EXTRACTION
# =======================
df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")
df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

choice <- subset(df_train, select = c(Ch1, Ch2, Ch3, Ch4))
subset_train <- subset(df_train, select = -c(Task, Ch1, Ch2, Ch3, Ch4, educ, gender, region, segment, ppark, night, miles, Case, No, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4, Urb, income, age))
subset_test <- subset(df_test, select = -c(Task, Ch1, Ch2, Ch3, Ch4, educ, gender, region, segment, ppark, night, miles, Case, No, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4, Urb, income, age))

# =======================
# USEFUL FUNCTIONS
# =======================
compute_precision <- function(class, confusion_matrix){
    # Class             : your desired class or row/column number 
    # confusion_matrix  : a table object representing the confusion matrix
    #                     the predicted class are in the rows
    return (confusion_matrix[class, class] / sum(confusion_matrix[class, ]))
}


# ==============================================================================
# PART 1
# XGBOOST MARK 1.1.2 
# submission 6
# ==============================================================================

# -------------
#  K-FOLD CV 
# -------------
set.seed(12)
k <- 5
folds <- createFolds(subset1_train$Choice, k = k, list = TRUE)
cv_results <- data.frame()

xgb_params <- list(
    booster = "gbtree",
    eta = 0.07,
    max_depth = 5,
    gamma = 4,
    subsample = 1,
    colsample_bytree = 1,
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    num_class = 4
)

for (i in 1:k) {
    cat("Fold", i, "\n")
    
    train_indices <- unlist(folds[-i])
    val_indices <- folds[[i]]
    
    train_data <- subset_train[train_indices, ]
    val_data <- subset_train[val_indices, ]
    y_train <- as.numeric(train_data$Choice) - 1
    y_val <- as.integer(val_data$Choice) - 1
    X_train <- train_data %>% select(-Choice)
    X_val <- val_data %>% select(-Choice)
    
    xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
    xgb_val <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)
    watchlist <- list(train = xgb_train, val = xgb_val)
    
    xgb_model <- xgb.train(
        params = xgb_params,
        data = xgb_train,
        nrounds = 150,
        watchlist = watchlist,
        verbose = 1
    )
    
    eval_log <- xgb_model$evaluation_log
    last_eval_score <- eval_log[nrow(eval_log), ]
    cat("Fold", i, "Train Log Loss:", last_eval_score$train_mlogloss, "\n")
    cat("Fold", i, "Validation Log Loss:", last_eval_score$val_mlogloss, "\n")
    cv_results <- rbind(cv_results, last_eval_score)
}

# -------------
#  METRICS
# -------------
average_train_logloss <- mean(cv_results$train_mlogloss)
average_val_logloss <- mean(cv_results$val_mlogloss)
average_overfitting <- average_val_logloss - average_train_logloss
cat("Average Training Log Loss:", average_train_logloss, "\n")
cat("Average Validation Log Loss:", average_val_logloss, "\n")
cat("Average Overfitting:", average_overfitting, "\n")  

# -------------------
#  TRAIN PREDICTIONS
# -------------------
X_train <- subset_train %>% select(-Choice)
y_train <- as.numeric(subset_train$Choice) 
preds_train_xgb <- predict(xgb_model, as.matrix(X_train), reshape = TRUE)
preds_train_xgb <- as.data.frame(preds_train_xgb)
colnames(preds_train_xgb) <- c("1","2","3","4")
preds_train_xgb$PredictedChoice <- apply(preds_train_xgb, 1, function(y) colnames(preds_train_xgb)[which.max(y)])

CM_xgb <- table(preds_train_xgb$PredictedChoice, y_train)
XGB_precision_1 <- compute_precision(1, CM_xgb)
XGB_precision_2 <- compute_precision(2, CM_xgb)
XGB_precision_3 <- compute_precision(3, CM_xgb)
XGB_precision_4 <- compute_precision(4, CM_xgb)
XGB_precision <- c(XGB_precision_1, XGB_precision_2, XGB_precision_3, XGB_precision_4)
XGB_precision

compute_TP <- function(class, CM){
    return (CM[class,class])
}
compute_FP <- function(class, CM){
    
}


xgb_TP <- CM_xgb[1,1]+CM_xgb[2,2]+CM_xgb[3,3]+CM_xgb[4,4]
xgb_FN <- (CM_xgb[2,1]+CM_xgb[3,1]+CM_xgb[4,1]
           +CM_xgb[1,2]+CM_xgb[3,2]+CM_xgb[4,2]
           +CM_xgb[1,3]+CM_xgb[2,3]+CM_xgb[4,3]
           +CM_xgb[1,4]+CM_xgb[2,4]+CM_xgb[3,4])
xgb_FP <- (CM_xgb[1,2]+CM_xgb[1,3]+CM_xgb[1,4]
           +CM_xgb[2,1]+CM_xgb[2,3]+CM_xgb[2,4]
           +CM_xgb[3,1]+CM_xgb[3,2]+CM_xgb[3,4]
           +CM_xgb[4,1]+CM_xgb[4,2]+CM_xgb[4,3])
xgb_TN <- 
xgb_TPR <- (xgb_TP) / (xgb_FN + xgb_TP) 
xgb_TPR
xgb_FNR <- (xgb_FN) / (xgb_FN + xgb_TP) 


# preds_train_xgb$correct <- ifelse(preds_train_xgb$PredictedChoice == y_train, 1, 0) 
# preds_train_xgb$ActualChoice <- y_train
# wrong_predictions <- subset(preds_train_xgb, correct == 0)
# table(wrong_predictions$PredictedChoice, wrong_predictions$ActualChoice)
















# ==============================================================================
# PART 2
# MNL ONE HOT ENCODING
# multinomial_logit_hot_encoding.R
# ==============================================================================
# MULTINOMIAL LOGIT MODEL
library(mlogit)
library(caret)

# DATA EXTRACTION 
train_raw <- read.csv("train2024.csv")
test_raw <- read.csv("test2024.csv")
train_raw$Choice <- ifelse(train_raw$Ch1 == 1, 1, ifelse(train_raw$Ch2 == 1, 2, ifelse(train_raw$Ch3 == 1, 3, 4)))
test_raw$Choice <- sample(c(1, 2, 3, 4), nrow(test_raw), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

id_train <- subset(train_raw, select = c(Case, Task, No, Ch1, Ch2, Ch3, Ch4))
id_test <- subset(test_raw, select = c(Case, Task, No, Ch1, Ch2, Ch3, Ch4))

df_train <- subset(train_raw, select = -c(segment,segmentind,year,yearind,milesa,miles,milesind,nighta,night,
                                          nightind,ppark,pparkind,gender,genderind,age,ageind,agea,educ,
                                          educind,region,regionind,Urb,Urbind,income,incomeind,incomea,
                                          Ch1,Ch2,Ch3,Ch4))
df_test <- subset(test_raw, select = -c(segment,segmentind,year,yearind,milesa,miles,milesind,nighta,night,
                                        nightind,ppark,pparkind,gender,genderind,age,ageind,agea,educ,
                                        educind,region,regionind,Urb,Urbind,income,incomeind,incomea,
                                        Ch1,Ch2,Ch3,Ch4))

# ----------------------------------------------------------------------------------------------------
# FEATURE ENGINEERING (ONE HOT ENCODING)

# PART 1. ONE HOT ENCODING SAFETY PACKAGE FEATURES

# 1. Converting the car package features into factors / categorical variables
car_features <- c('CC1','GN1','NS1','BU1','FA1','LD1','BZ1','FC1','FP1','RP1','PP1','KA1','SC1','TS1','NV1','MA1','LB1','AF1','HU1','Price1',
                  'CC2','GN2','NS2','BU2','FA2','LD2','BZ2','FC2','FP2','RP2','PP2','KA2','SC2','TS2','NV2','MA2','LB2','AF2','HU2','Price2',
                  'CC3','GN3','NS3','BU3','FA3','LD3','BZ3','FC3','FP3','RP3','PP3','KA3','SC3','TS3','NV3','MA3','LB3','AF3','HU3','Price3',
                  'CC4','GN4','NS4','BU4','FA4','LD4','BZ4','FC4','FP4','RP4','PP4','KA4','SC4','TS4','NV4','MA4','LB4','AF4','HU4','Price4')
for (car_feature in car_features){
    df_train[, car_feature] <- factor(df_train[, car_feature])
    df_test[, car_feature] <- factor(df_test[, car_feature])
}

# # 2. Check and remove factors with only one level
for (car_feature in car_features) {
    if (length(unique(df_train[, car_feature])) <= 1) {
        df_train[, car_feature] <- NULL
        df_test[, car_feature] <- NULL
    }
}

# 3. Create dummy variables
dummy <- dummyVars(" ~ .", data = df_train)
train_data <- data.frame(predict(dummy, newdata = df_train))
test_data <- data.frame(predict(dummy, newdata = df_test))

# 4. Swap column names from FeatureAlt.Level to FeatureLevel.Alt
swap_xy <- function(col_name) {
    sub("([A-Za-z]+)([0-9]+)\\.([0-9]+)", "\\1\\3.\\2", col_name)
}

train_data_tochange <- subset(train_data,select = -c(Choice,Case,No,Task))
colnames(train_data_tochange) <- sapply(colnames(train_data_tochange), swap_xy)

test_data_tochange <- subset(test_data,select = -c(Choice,Case,No,Task))
colnames(test_data_tochange) <- sapply(colnames(test_data_tochange), swap_xy)

# 5. Add the columns for the 4th alternative.
# Step 1: Extract the PrefixY part from current column names
prefix_y <- sub("([A-Za-z]+[0-9]+)\\.[0-9]+", "\\1", colnames(train_data_tochange)[1:91])

# Step 2: Create new column names by appending ".4"
new_colnames <- paste0(prefix_y, ".4")

# Step 3: Add new columns with these names to the data frame
# Initialize the new columns with NA or any other value as required
for (new_col in new_colnames) {
    train_data_tochange[[new_col]] <- 0
    test_data_tochange[[new_col]] <- 0
}

# 6. Remove 1 indicator variable for each category to reduce model variability
train_features <- subset(train_data_tochange, select = -c(AF0.1, AF0.2, AF0.3, AF0.4,
                                                          BU0.1, BU0.2, BU0.3, BU0.4,
                                                          BZ0.1, BZ0.2, BZ0.3, BZ0.4,
                                                          CC0.1, CC0.2, CC0.3, CC0.4, 
                                                          FA0.1, FA0.2, FA0.3, FA0.4,
                                                          FC0.1, FC0.2, FC0.3, FC0.4,
                                                          FP0.1, FP0.2, FP0.3, FP0.4,
                                                          GN0.1, GN0.2, GN0.3, GN0.4,
                                                          HU0.1, HU0.2, HU0.3, HU0.4,
                                                          KA0.1, KA0.2, KA0.3, KA0.4,
                                                          LB0.1, LB0.2, LB0.3, LB0.4,
                                                          LD0.1, LD0.2, LD0.3, LD0.4,
                                                          MA0.1, MA0.2, MA0.3, MA0.4,
                                                          NS0.1, NS0.2, NS0.3, NS0.4,
                                                          NV0.1, NV0.2, NV0.3, NV0.4,
                                                          PP0.1, PP0.2, PP0.3, PP0.4,
                                                          RP0.1, RP0.2, RP0.3, RP0.4,
                                                          SC0.1, SC0.2, SC0.3, SC0.4,
                                                          TS0.1, TS0.2, TS0.3, TS0.4,
                                                          Price1.1, Price1.2, Price1.3, Price1.4))

test_features <- subset(test_data_tochange, select = -c(AF0.1, AF0.2, AF0.3, AF0.4,
                                                        BU0.1, BU0.2, BU0.3, BU0.4,
                                                        BZ0.1, BZ0.2, BZ0.3, BZ0.4,
                                                        CC0.1, CC0.2, CC0.3, CC0.4, 
                                                        FA0.1, FA0.2, FA0.3, FA0.4,
                                                        FC0.1, FC0.2, FC0.3, FC0.4,
                                                        FP0.1, FP0.2, FP0.3, FP0.4,
                                                        GN0.1, GN0.2, GN0.3, GN0.4,
                                                        HU0.1, HU0.2, HU0.3, HU0.4,
                                                        KA0.1, KA0.2, KA0.3, KA0.4,
                                                        LB0.1, LB0.2, LB0.3, LB0.4,
                                                        LD0.1, LD0.2, LD0.3, LD0.4,
                                                        MA0.1, MA0.2, MA0.3, MA0.4,
                                                        NS0.1, NS0.2, NS0.3, NS0.4,
                                                        NV0.1, NV0.2, NV0.3, NV0.4,
                                                        PP0.1, PP0.2, PP0.3, PP0.4,
                                                        RP0.1, RP0.2, RP0.3, RP0.4,
                                                        SC0.1, SC0.2, SC0.3, SC0.4,
                                                        TS0.1, TS0.2, TS0.3, TS0.4,
                                                        Price1.1, Price1.2, Price1.3, Price1.4))
# 7. Add Case, No, Task, Choice columns back 
train_features$No <- seq(1, nrow(train_features))
train <- merge(id_train, train_features, on='No')
train["Choice"] <- train_data["Choice"]

test_features$No <- seq(21566, 21566+nrow(test_features)-1)
test <- merge(id_test, test_features, on='No')
test["Choice"] <- test_data["Choice"]

# 8. Clearing the environment 
rm(train_data, train_data_tochange, train_features,
   test_data, test_data_tochange, test_features,
   dummy)

# ------------------------------------------------------------------------
# Part 2 (1 HOT ENCODE HUMAN FEATURES)

# 1. Select all human features to encode

train_human_to_encode <- subset(train_raw, select = c(No,segmentind, yearind, milesind, nightind,
                                                      pparkind, genderind, ageind, educind,
                                                      regionind, Urbind, incomeind))

test_human_to_encode <- subset(test_raw, select = c(No,segmentind, yearind, milesind, nightind,
                                                    pparkind, genderind, ageind, educind,
                                                    regionind, Urbind, incomeind))

# 2. Convert all human features to factors
human_features <- c('segmentind', 'yearind', 'milesind', 'nightind',
                    'pparkind', 'genderind', 'ageind', 'educind',
                    'regionind', 'Urbind', 'incomeind')

for (human_feature in human_features){
    train_human_to_encode[, human_feature] <- factor(train_human_to_encode[, human_feature])
    test_human_to_encode[, human_feature] <- factor(test_human_to_encode[, human_feature])
}

# 3. Check and remove factors with only one level
for (human_feature in human_features) {
    if (length(unique(train_human_to_encode[, human_feature])) <= 1) {
        train_human_to_encode[, human_feature] <- NULL
        test_human_to_encode[, human_feature] <- NULL
    }
}                                

# 4. Create dummy variables
dummy <- dummyVars(" ~ .", data = train_human_to_encode)
train_human <- data.frame(predict(dummy, newdata = train_human_to_encode))
test_human <- data.frame(predict(dummy, newdata = test_human_to_encode))

# 5. Remove 1 indicator variable for each category to reduce model variability
train_human <- subset(train_human, select = -c(segmentind.1, yearind.1, milesind.1, nightind.1,
                                               pparkind.1, genderind.1, ageind.1, educind.1,
                                               regionind.1, Urbind.1, incomeind.1))

test_human <- subset(test_human, select = -c(segmentind.1, yearind.1, milesind.1, nightind.1,
                                             pparkind.1, genderind.1, ageind.1, educind.1,
                                             regionind.1, Urbind.1, incomeind.1))
# 6. Merge with Part 1.

train <- merge(train, train_human, on='No')
test <- merge(test, test_human, on='No')

# ------------------------------------------------------------------------
# Merge data with incomea, nighta, agea
human_train <- subset(train_raw, select = c(agea,nighta,incomea,No))
human_test <- subset(test_raw,select = c(agea,nighta,incomea,No))

train <- merge(train, human_train, on='No')
test <- merge(test, human_test, on = 'No')

rm(test_human,test_human_to_encode,train_human,train_human_to_encode,dummy,human_test,human_train)

# ------------------------------------------------------------------------
# PART 3 MODEL BUILDING

# Formatting data for mlogit()

S_train <- dfidx(subset(train, Task <=14), shape="wide", choice="Choice", sep=".",
                 varying = c(8:291), idx = c("No", "Case"))
S_val <- dfidx(subset(train, Task > 14), shape="wide", choice="Choice", sep=".",
               varying = c(8:291), idx = c("No", "Case"))
S_test <- dfidx(test, shape="wide", choice="Choice", sep=".",
                varying = c(8:291), idx = c("No", "Case"))
S_everything <- dfidx(train, shape="wide", choice="Choice", sep=".", 
                      varying = c(8:291), idx = c("No", "Case"))

MNL_2 <- mlogit(Choice~AF1+AF2+BU1+BU2+BU3+BU4+BU5+BZ1+BZ2+BZ3+CC1+CC2+CC3+
                    FA1+FC1+FP1+FP2+FP3+HU1+KA1+KA2+LB3+LD1+LD2+MA1+MA3+NS1+
                    NS4+NV1+NV2+PP1+PP2+RP1+SC1+SC2+SC3+TS1+TS2+Price2+Price3+
                    Price4+Price5+Price6+Price7+Price8+Price9+Price10+Price11+Price12-1,
                    data=S_train)
preds_train_MNL <- predict(MNL_2,newdata = S_everything)
preds_train_MNL <- as.data.frame(preds_train_MNL)
colnames(preds_train_MNL) <- c("1","2","3","4")
preds_train_MNL$Choice <- apply(preds_train_MNL, 1, function(y) colnames(preds_train_MNL)[which.max(y)])
y_train_MNL <- train[4:7]
colnames(y_train_MNL) <- c("1","2","3","4")
y_train_MNL$Choice <- apply(y_train_MNL, 1, function(y) colnames(y_train_MNL)[which.max(y)])


CM_MNL_onehot <- table(preds_train_MNL$Choice, y_train_MNL$Choice)
MNL_onehot_precision_1 <- compute_precision(1, CM_MNL_onehot)
MNL_onehot_precision_2 <- compute_precision(2, CM_MNL_onehot)
MNL_onehot_precision_3 <- compute_precision(3, CM_MNL_onehot)
MNL_onehot_precision_4 <- compute_precision(4, CM_MNL_onehot)
MNL_onehot_precision <- c(MNL_onehot_precision_1, MNL_onehot_precision_2, MNL_onehot_precision_3, MNL_onehot_precision_4)
MNL_onehot_precision





























