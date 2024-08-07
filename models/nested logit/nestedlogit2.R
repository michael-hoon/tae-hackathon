library(mlogit)
library(caret)

# DATA EXTRACTION 
train_raw <- read.csv("/home/micha/TAE_Code/tae-hackathon/data/train2024.csv")
test_raw <- read.csv("/home/micha/TAE_Code/tae-hackathon/data/test2024.csv")
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


# ------------------------------------------------------------------------
# PART 3 MODEL BUILDING

# Formatting data for mlogit()

S_train <- dfidx(subset(train, Task <=14), shape="wide", choice="Choice", sep=".",
                 varying = c(8:291), idx = c("No", "Case"))

S_val <- dfidx(subset(train, Task > 14), shape="wide", choice="Choice", sep=".",
               varying = c(8:291), idx = c("No", "Case"))

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
# S_train <- dfidx(subset(train), shape="wide", choice="Choice", varying =c(4:83), sep="", idx = list(c("No", "Case")))
# S_test <- dfidx(subset(test), shape="wide", choice="Choice", varying = c(4:83), sep="", idx = list(c("No", "Case")))
# nested logit model

# first set a list of nests corresponding to the first 3 choices from the consumer being the first nest, and choice 4 being the second nest

# nests <- list(
    # Choice = c("Ch1", "Ch2","Ch3"),
    # NoChoice = c("Ch4")
# )

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


# Compute accuracy
compute_accuracy <- function(P, choice){
  # P: predicted choices
  # choice: actual choices
  # Determine the predicted choice by taking the column index of the max probability in each row
  predicted_choice <- apply(P, 1, which.max)
  
  # Determine the actual choice by taking the column index of the max probability in each row
  actual_choice <- apply(choice, 1, which.max)
  
  # Compute the confusion matrix
  confusion_matrix <- table(Predicted = predicted_choice, Actual = actual_choice)
  
  (sum(diag(confusion_matrix)))/(sum((confusion_matrix)))
}

# here, the nests should be set via the idx$id2 choice variable entries 1 2 3 4, after transformation to wide format via dfidx
nests <- list(
    nest1 = c("1", "2", "3"),
    nest2 = c("4")
)

nestlogit <- mlogit(
    formula = Choice ~ AF1+AF2+BU1+BU2+BU3+BU4+BU5+BZ1+BZ2+CC1+CC2+CC3+
                +FA1+FC1+FP1+FP2+FP3+GN1+HU1+KA1+KA2+
                +LB3+LD1+LD2+MA1+MA3+NS1+NS4+NV1+NV2+
                +PP1+PP2+RP1+SC1+SC2+SC3+TS1+TS2+Price2+Price3+
                +Price4+Price5+Price6+Price7+Price8+Price9+Price10+Price11+Price12-1|
                +Urbind.3+ageind.2+ageind.3+ageind.4+ageind.5+
                +educind.2+genderind.2+incomeind.4+incomeind.8+incomeind.13+incomeind.14+incomeind.16+
                +incomeind.19+incomeind.28+milesind.2+milesind.3+milesind.4+nightind.5+nightind.7+
                +nightind.8+nightind.10+pparkind.2+pparkind.3+pparkind.4+pparkind.5+regionind.2+
                +regionind.3+regionind.4+segmentind.3+segmentind.5+segmentind.6+yearind.3+
                +yearind.4-1,
    data = S_train,
    nests = nests,
    un.nest.el = T
)

summary(nestlogit)


train_predictions <- predict(nestlogit, newdata=S_train)
test_predictions <- predict(nestlogit, newdata=S_val)

actual_trainprobs <- subset(train,Task<=14)[4:7]
actual_valprobs <- subset(train,Task>14)[4:7]


compute_logloss(train_predictions , actual_trainprobs)
compute_accuracy(train_predictions , actual_trainprobs)
compute_logloss(test_predictions , actual_valprobs)
compute_accuracy(test_predictions , actual_valprobs)
