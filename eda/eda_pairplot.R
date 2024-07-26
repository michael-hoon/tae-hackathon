setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\eda")

df_train <- read.csv("train2024.csv")
df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))

# features <- c("Price1", "Price2", "Price3", "Price4")
# features <- c('Urbind', 'agea', 'educind', 'genderind', 'incomea', 'milesa', 'nighta', 'pparkind', 'regionind', 'segmentind', 'year')
# df_plot <- subset(df_train, select=features)
# plot(df_plot)


# ==========================================
# # GGally Implementation
# https://r-charts.com/correlation/ggpairs/
# ==========================================
library(GGally)
library(dplyr)
# # ggpairs(iris,                 # Data frame
# #         columns = 1:4,        # Columns
# #         aes(color = Species,  # Color by group (cat. variable)
# #             alpha = 0.5))     # Transparencyprice_1 <- which(names(df_train) == "Price1")

# Prices
price_1 <- which(names(df_train) == "Price1")
price_2 <- which(names(df_train) == "Price2")
price_3 <- which(names(df_train) == "Price3")
price_4 <- which(names(df_train) == "Price4")
features <- c(price_1, price_2, price_3, price_4)
ggpairs(df_train, 
        columns = features,
        aes(color = gender,
            alpha=0.5))

# Choices
choice_1 <- which(names(df_train) == "Ch1")
choice_2 <- which(names(df_train) == "Ch2")
choice_3 <- which(names(df_train) == "Ch3")
choice_4 <- which(names(df_train) == "Ch4")
features <- c(choice_1, choice_2, choice_3, choice_4)
ggpairs(df_train, 
        columns = features,
        aes(color = gender,
            alpha=0.5))

# LB x Choices
# LB: Low speed breaking assist
df_train$Choice <- as.factor(df_train$Choice)
features <- c("LB1", "LB2", "LB3", "LB4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:length(df_selected),
        aes(color = Choice,
            alpha=0.5))

# income x Choices
features <- c("incomea", "Ch1", "Ch2", "Ch3", "Ch4", "Choice")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:(length(df_selected)-1),
        aes(color = Choice, alpha = 0.5),
        lower = list(continuous = wrap("points", position = position_jitter(width = 0.2, height = 0.2))))

# LB x Price x Choices
# LB: Low speed breaking assist
features <- c("LB1", "LB2", "LB3", "LB4", "Choice", "gender")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:length(df_selected),
        aes(color = gender,
            alpha=0.5))


# People variables
features <- c("segment", "Urb", "milesa", "nighta", "agea", "incomea", "gender")
df_selected <- df_train[, features]
ggpairs(df_selected, 
        columns = 1:length(df_selected),
        aes(color = gender,
            alpha=0.5))






