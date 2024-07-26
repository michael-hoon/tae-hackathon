setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\eda")
library(ggplot2)
library(GGally)

df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")

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









