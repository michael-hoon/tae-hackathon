#Feature Selection
library(xgboost)
library(caTools)
library(dplyr)
library(caret)
library(devtools)
library(FeatureSelection)

setwd("C:/Users/User/OneDrive/Desktop/TAE Comp 2024")
df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")

df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

# FEATURE SELECTION
choice <- subset(df_train, select = c(Ch1, Ch2, Ch3, Ch4))
indicators <- subset(df_train, select = -c(Task, Ch1, Ch2, Ch3, Ch4,No,educ,gender,region,segment,ppark,night,miles,Case, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4,Urb,income,age, Choice))

params_xgboost <- list(params = list(
  "booster" = "gbtree",
  "eta" = 0.01,
  "max_depth" = 8,
  "gamma" = 4,
  "subsample" = 0.75,
  "colsample_bytree" = 1,
  "objective" = "multi:softprob",
  "eval_metric" = "mlogloss",
  "num_class" = 4), 
  nrounds = 5000,
  verbose = 1
  )

Y <- df_test$Choice - 1
sel1 <- feature_selection(indicators,
                          Y,
                          method ='xgboost', 
                          params_xgboost = params_xgboost,
                          CV_folds = 5)                   


str(sel1)
params_barplot = list(keep_features = 77, horiz = TRUE, cex.names = 1.0)
barplot_feat_select(sel1, params_barplot, xgb_sort = 'Gain')
sel1 <- sel1[order(sel1$Gain, decreasing = TRUE),]
sel1
