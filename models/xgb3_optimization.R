library(xgboost)
library(caTools)
library(dplyr)
library(caret)
library(tidymodels)

# DATA EXTRACTION
df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")
df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

# DATA SPLITTING
library(tidymodels)
set.seed(123)
choice <- subset(df_train, select = c(Ch1, Ch2, Ch3, Ch4))
train <- subset(df_train, select = -c(Task, Ch1, Ch2, Ch3, Ch4, educ, gender, region, segment, ppark, night, miles, Case, No, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4, Urb, income, age))
validation <- subset(df_test, select = -c(Task, Ch1, Ch2, Ch3, Ch4, educ, gender, region, segment, ppark, night, miles, Case, No, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4, Urb, income, age))

split <- train %>%
    initial_split(prop = 0.8, strata = Choice)

train <- training(split)
test <- testing(split)

# K-FOLD CROSS VALIDATION
set.seed(234)
folds <- vfold_cv(train, strata = Choice)

xgb_spec <-boost_tree(
    trees = 500,
    tree_depth = tune(), 
    min_n = tune(),
    loss_reduction = tune(),                    ## first three: model complexity
    sample_size = tune(), mtry = tune(),        ## randomness
    learn_rate = tune()                         ## step size
) %>%
    set_engine("xgboost") %>%
    set_mode("classification")
# xgb_spec

xgb_wf <- workflow() %>%
    add_formula(Churn ~.) %>%
    add_model(xgb_spec)
xgb_wf


xgb_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train),
    learn_rate(),
    size = 20
)

library(finetune)
doParallel::registerDoParallel()

set.seed(234)
xgb_res <-tune_grid(
    xgb_wf,
    resamples = folds,
    grid = xgb_grid,
    control = control_grid(save_pred  = TRUE)
)
xgb_res


# write.csv(xgb_preds, " ")
