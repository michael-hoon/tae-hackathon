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
df_train <- subset(train_raw, select = -c(segment,segmentind,year,yearind,milesa,miles,milesind,nighta,night,
                                          nightind,ppark,pparkind,gender,genderind,age,ageind,agea,educ,
                                          educind,region,regionind,Urb,Urbind,income,incomeind,incomea,
                                          Ch1,Ch2,Ch3,Ch4))
df_test <- subset(test_raw, select = -c(segment,segmentind,year,yearind,milesa,miles,milesind,nighta,night,
                                        nightind,ppark,pparkind,gender,genderind,age,ageind,agea,educ,
                                        educind,region,regionind,Urb,Urbind,income,incomeind,incomea,
                                        Ch1,Ch2,Ch3,Ch4))

# FEATURE ENGINEERING (ONE HOT ENCODING)
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

# 4. Swap column names from FeatureAlt.Level to FeatureLevel.Alt
swap_xy <- function(col_name) {
  sub("([A-Za-z]+)([0-9]+)\\.([0-9]+)", "\\1\\3.\\2", col_name)
}
train_data_tochange <- subset(train_data,select = -c(Choice,Case,No,Task))
colnames(train_data_tochange) <- sapply(colnames(train_data_tochange), swap_xy)

# 5. Add the columns for the 4th alternative.
    # Step 1: Extract the PrefixY part from current column names
    prefix_y <- sub("([A-Za-z]+[0-9]+)\\.[0-9]+", "\\1", colnames(train_data_tochange)[1:91])

    # Step 2: Create new column names by appending ".4"
    new_colnames <- paste0(prefix_y, ".4")

    # Step 3: Add new columns with these names to the data frame
    # Initialize the new columns with NA or any other value as required
    for (new_col in new_colnames) {
      train_data_tochange[[new_col]] <- 0
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

# 7. Add Case, No, Task, Choice columns back 
train_features$No <- seq(1, nrow(train_features))
train <- merge(id_train, train_features, on='No')
train["Choice"] <- train_data["Choice"]

# ------------------------------------------------------------------------


S_train <- dfidx(subset(train, Task <=12), shape="wide", choice="Choice", sep=".",
                 varying = c(8:291), idx = c("No", "Case"))

# MNL model
M <- mlogit(Choice~AF1+AF2+AF3+BU1+BU2+BU3+BU4+BU5+BU6+BZ1+BZ2+BZ3+CC1+CC2+CC3+
              FA1+FA2+FC1+FC2+FP1+FP2+FP3+FP4+GN1+GN2+HU1+HU2+KA1+KA2+LB1+LB2+
              LB3+LB4+LD1+LD2+LD3+MA1+MA2+MA3+MA4+NS1+NS2+NS3+NS4+NS5+NV1+NV2+
              NV3+PP1+PP2+PP3+RP1+RP2+SC1+SC2+SC3+SC4+TS1+TS2+TS3+Price2+Price3+
              Price4+Price5+Price6+Price7+Price8+Price9+Price10+Price11+Price12-1, data=S_train)

# Mixed logit model karthik1 BAD!
M <- mlogit(Choice~CC1+CC2+CC3+NS1+NS4+BU1+BU2+BU3+BU4+BU5+FA1+LD1+LD2+BZ1+BZ2+
              FC1+FP1+FP2+FP3+RP1-1,
              rpar = c(BZ1='n',BZ2='n',FP1='n',FP2='n',FP3='n',
                       RP1='n'), data=S_train)

# Mixed logit with karthik
M <- mlogit(Choice~PP1+PP2+KA1+KA2+SC1+SC2+SC3+TS1+TS2+TS3+NV1+NV2+MA3+AF1+AF2+
                   HU1+Price2+Price3+Price4+Price5+Price6+Price7+Price8+Price9+
                   Price10+Price11+Price12-1, 
            rpar=c(BZ='n', FP='n', RP='n',PP='n',NV='n') data = S_train)
summary(M)
-M$logLik/nrow(subset(train, Task <=12))



S_val <- dfidx(subset(train, Task > 12), shape="wide", choice="Choice", sep=".",
               varying = c(8:291), idx = c("No", "Case"))
pred_probs <- predict(M,newdata = S_val)

real_probs <- subset(train,Task>12)[4:7]

compute_logloss(pred_probs, real_probs)





#Compute logloss function
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












