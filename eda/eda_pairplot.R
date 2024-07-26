setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\eda")

df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")

# features <- c("Price1", "Price2", "Price3", "Price4")
features <- c('Urbind', 'agea', 'educind', 'genderind', 'incomea', 'milesa', 'nighta', 'pparkind', 'regionind', 'segmentind', 'year')

df_plot <- subset(df_train, select=features)
plot(df_plot)
