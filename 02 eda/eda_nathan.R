setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\eda")
library(ggplot2)
library(GGally)

df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")
df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))

# SUMMARY
str(df_train)
summary(df_train)

custom_jitter <- function(data, mapping, ...) {
    ggplot(data, mapping) +
        geom_point(position = position_jitter(width = 0.1, height = 0.1), alpha = 0.6, size = 2, ...) +
        theme_minimal()
}

ggpairs(df_train,
        columns = c('Urbind', 'agea', 'educind', 'genderind', 'gender', 'incomea', 'milesa', 'nighta', 'pparkind', 'regionind', 'segmentind', 'year'),
        upper = list(continuous = custom_jitter),
        lower = list(continuous = custom_jitter),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5))
)

# NEW
df_train$Choice <- as.factor(df_train$Choice)
choose_1 <- subset(df_train, df_train$Ch1==1)
features <- c("LB1", "LB2", "LB3", "LB4", "Choice")
df_selected <- choose_1[, features]
ggpairs(df_selected, 
        columns = 1:length(df_selected),
        aes(color = Choice,
            alpha=0.5))

# NEW AGAIN
features <- c("LB1", "LB2", "LB3", "LB4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

ggpairs(df_train,
        columns = features,
        upper = list(continuous = custom_jitter),
        lower = list(continuous = custom_jitter),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5))
)


features <- c("AF1", "AF2", "AF3", "AF4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("BU1", "BU2", "BU3", "BU4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("BZ1", "BZ2", "BZ3", "BZ4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("CC1", "CC2", "CC3", "CC4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("FA1", "FA2", "FA3", "FA4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("FC1", "FC2", "FC3", "FC4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("FP1", "FP2", "FP3", "FP4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("GN1", "GN2", "GN3", "GN4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))

features <- c("GN1", "GN2", "GN3", "GN4", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = "Choice",
            alpha=0.5))


# SCATTERPLOTS PER CHOICE
features <- c("GN1", "GN2", "GN3", "GN4", "Choice")
df_selected <- df_train[, features]
df_selected <- subset(df_selected, Choice == 1)
# ggpairs(df_selected, columns = 1:(length(df_selected)-1), 
#         upper = list(continuous = custom_jitter), 
#         lower = list(continuous = custom_jitter), 
#         diag = list(continuous = wrap("densityDiag", alpha = 0.5)))

table(df_selected$GN1)
table(df_selected$GN2)
table(df_selected$GN3)
table(df_selected$GN4)







