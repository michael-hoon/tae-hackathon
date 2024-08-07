setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
# MULTINOMIAL LOGIT MODEL
# Here, car features are treated as categorical variables with one-hot encoding 
library(mlogit)
library(caret)

# DATA EXTRACTION 
train_raw <- read.csv("train2024.csv")
test_raw <- read.csv("test2024.csv")
train_raw$Choice <- ifelse(train_raw$Ch1 == 1, 1, ifelse(train_raw$Ch2 == 1, 2, ifelse(train_raw$Ch3 == 1, 3, 4)))
test_raw$Choice <- sample(c(1, 2, 3, 4), nrow(test_raw), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

id_train <- subset(train_raw, select = c(Case, Task, No, Ch1, Ch2, Ch3, Ch4))
df_train <- subset(train_raw, select = -c(segmentind,year,miles,milesind,night,nightind,
                                          pparkind,genderind,age,ageind,educind,regionind,
                                          Urbind,income,incomeind,Ch1,Ch2,Ch3,Ch4))
df_test <- subset(test_raw, select = -c(segmentind,year,miles,milesind,night,nightind,
                                        pparkind,genderind,age,ageind,educind,regionind,
                                        Urbind,income,incomeind,Ch1,Ch2,Ch3,Ch4))

# FEATURE ENGINEERING
# 1. Converting the car features into factors / categorical variables
car_features <- c('CC1','GN1','NS1','BU1','FA1','LD1','BZ1','FC1','FP1','RP1','PP1','KA1','SC1','TS1','NV1','MA1','LB1','AF1','HU1',
                  'CC2','GN2','NS2','BU2','FA2','LD2','BZ2','FC2','FP2','RP2','PP2','KA2','SC2','TS2','NV2','MA2','LB2','AF2','HU2',
                  'CC3','GN3','NS3','BU3','FA3','LD3','BZ3','FC3','FP3','RP3','PP3','KA3','SC3','TS3','NV3','MA3','LB3','AF3','HU3')
                  # 'CC4','GN4','NS4','BU4','FA4','LD4','BZ4','FC4','FP4','RP4','PP4','KA4','SC4','TS4','NV4','MA4','LB4','AF4','HU4')
for (car_feature in car_features){
  df_train[, car_feature] <- factor(df_train[, car_feature])
  df_test[, car_feature] <- factor(df_test[, car_feature])
}

# # 2. Check and remove factors with only one level
# for (car_feature in car_features) {
#   if (length(unique(df_train[, car_feature])) <= 1) {
#     df_train[, car_feature] <- NULL
#     df_test[, car_feature] <- NULL
#   }
# }

# 3. Create dummy variables
dummy <- dummyVars(" ~ .", data = df_train)
train_data <- data.frame(predict(dummy, newdata = df_train))

# 4. Remove 1 indicator variable for each category to reduce model variability
train_features <- subset(train_data, select = -c(AF1.0, AF2.0, AF3.0, BU1.0, BU2.0, BU3.0, 
                                     BZ1.0, BZ2.0, BZ3.0, CC1.0, CC2.0, CC3.0, 
                                     FA1.0, FA2.0, FA3.0, FC1.0, FC2.0, FC3.0, 
                                     FP1.0, FP2.0, FP3.0, GN1.0, GN2.0, GN3.0, 
                                     HU1.0, HU2.0, HU3.0, KA1.0, KA2.0, KA3.0, 
                                     LB1.0, LB2.0, LB3.0, LD1.0, LD2.0, LD3.0, 
                                     MA1.0, MA2.0, MA3.0, NS1.0, NS2.0, NS3.0, 
                                     NV1.0, NV2.0, NV3.0, PP1.0, PP2.0, PP3.0, 
                                     RP1.0, RP2.0, RP3.0, SC1.0, SC2.0, SC3.0, 
                                     TS1.0, TS2.0, TS3.0, segmentSmall.Car, pparkNever, 
                                     genderFemale, educTrade.Vocational.School, regionW, UrbSuburban))

# 5. Remove irrelevant features / redundant columns 
train_features <- subset(train_features, select = -c(Case, No, Task)) 
train_features$No <- seq(1, nrow(train_features))
train <- merge(id_train, train_features, on='No')

S_train <- dfidx(subset(train, Task <=12), shape="wide", choice="Choice", sep="", varying = c(8:191), idx = c("No", "Case"))
M <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train)
summary(M)
M <- mlogit(Choice~CC+GN+BU+FA+LD+BZ+FC+RP+PP+KA+SC+TS+NV+MA+LB+HU+Price-1, data=S_train)
summary(M)
















