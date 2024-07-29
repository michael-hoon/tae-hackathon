setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
# MIXED LOGIT MODEL

# DATA EXTRACTION
safety <- read.csv("train2024.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, ifelse(safety$Ch2 == 1, 2, ifelse(safety$Ch3 == 1, 3, 4)))
str(safety)
summary(safety)
head(safety)

# DATA SPLITTING # 1
set.seed(12)
split <- sample(1:nrow(safety), 0.8*nrow(safety))
train <- safety[split,] # actual train 80%
test <- safety[-split,] # actually this is validation set 20%
train_choice <- subset(safety, select = c(Ch1,Ch2,Ch3,Ch4))[split,]
val_choice <- subset(safety, select = c(Ch1,Ch2,Ch3,Ch4))[-split,] # this is y_ij for logloss

# DATA SPLITTING # 2
train <- subset(safety, Task <= 12)
test <- subset(safety, Task > 12)
train_choice <- subset(train, select = c(Ch1,Ch2,Ch3,Ch4)) 
val_choice <- subset(test, select = c(Ch1,Ch2,Ch3,Ch4)) # this is y_ij for logloss

# ========================
#    MODEL TRAINING # 1 
# ========================
library(mlogit)
S_train <- dfidx(subset(train), shape="wide", choice="Choice", varying =c(4:83), sep="",idx = list(c("No", "Case")))

M1 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train,
             rpar=c(CC='n', GN='n', NS='n', BU='n',FA='n',LD='n',BZ='n',FC='n',FP='n',RP='n',PP='n',KA='n',
                    SC='n',TS='n',NV='n',MA='n',LB='n',AF='n',HU='n',Price='n'),
             panel = TRUE, print.level=TRUE)
summary(M1)
M1_train_logloss <- -summary(M1)$logLik / nrow(S_train)
M1_train_logloss # 0.2575982

# df2 <- train 
# features <- c('CC1', 'GN1', 'NS1', 'BU1', 'FA1', 'LD1', 'BZ1', 'FC1', 'FP1', 'RP1', 
#               'PP1', 'KA1', 'SC1', 'TS1', 'NV1', 'MA1', 'LB1', 'AF1', 'HU1', 'Price1', 
#               'CC2', 'GN2', 'NS2', 'BU2', 'FA2', 'LD2', 'BZ2', 'FC2', 'FP2', 'RP2', 
#               'PP2', 'KA2', 'SC2', 'TS2', 'NV2', 'MA2', 'LB2', 'AF2', 'HU2', 'Price2', 
#               'CC3', 'GN3', 'NS3', 'BU3', 'FA3', 'LD3', 'BZ3', 'FC3', 'FP3', 'RP3', 
#               'PP3', 'KA3', 'SC3', 'TS3', 'NV3', 'MA3', 'LB3', 'AF3', 'HU3', 'Price3')
# df2 <- df2 %>% mutate(across(all_of(features), ~replace(., . == 0, 0.1))) 
# # since i am trying lognormal, the values of the variables must not be 0. 
# # hence we replace 0 with a small number 
# # it did NOT work out
# S_train_2 <- dfidx(subset(df2), shape="wide", choice="Choice", varying =c(4:83), sep="",idx = list(c("No", "Case")))
# 
# M2 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train_2,
#              rpar=c(CC='ln', GN='ln', NS='ln', BU='ln',FA='ln',LD='ln',BZ='ln',FC='ln',FP='ln',RP='ln',PP='ln',KA='ln',
#                     SC='ln',TS='ln',NV='ln',MA='ln',LB='ln',AF='ln',HU='ln',Price='ln'),
#              panel = TRUE, print.level=TRUE)
# summary(M2)

M3 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(CC='t', GN='t', NS='t', BU='t',FA='t',LD='t',BZ='t',FC='t',FP='t',RP='t',PP='t',KA='t',
                    SC='t',TS='t',NV='t',MA='t',LB='t',AF='t',HU='t',Price='t'), 
             panel = TRUE, print.level=TRUE)
summary(M3)
M3_train_logloss <- -summary(M3)$logLik / nrow(S_train)
M3_train_logloss # 0.2589319 

M4 <- mlogit(Choice~CC+GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1, data=S_train, 
             rpar=c(CC='u', GN='u', NS='u', BU='u',FA='u',LD='u',BZ='u',FC='u',FP='u',RP='u',PP='u',KA='u',
                    SC='u',TS='u',NV='u',MA='u',LB='u',AF='u',HU='u',Price='u'), 
             panel = TRUE, print.level=TRUE)
summary(M4)
M4_train_logloss <- -summary(M4)$logLik / nrow(S_train)
M4_train_logloss # 0.2595637 

# ========================
#     MODEL TESTING # 1
# ========================
T1 <- predict(M1, newdata=S_train) # train predictions
T3 <- predict(M3, newdata=S_train)
T4 <- predict(M4, newdata=S_train)

S_test <- dfidx(subset(test), shape="wide", choice="Choice", sep="", varying = c(4:83), idx = list(c("No", "Case")))
P1 <- predict(M1, newdata=S_test) # validation predictions
P3 <- predict(M3, newdata=S_test)
P4 <- predict(M4, newdata=S_test)

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

M1_train_logloss <- compute_logloss(T1, train_choice)
M3_train_logloss <- compute_logloss(T3, train_choice)
M4_train_logloss <- compute_logloss(T4, train_choice)
M1_train_logloss 
M3_train_logloss 
M4_train_logloss 

M1_val_logloss <- compute_logloss(P1, val_choice)
M3_val_logloss <- compute_logloss(P3, val_choice)
M4_val_logloss <- compute_logloss(P4, val_choice)
M1_val_logloss
M3_val_logloss
M4_val_logloss


# ========================
#     MODEL ANALYSIS # 1
# ========================
summary(M1)

# # Function to simulate the random parameters based on the estimated coefficients and their standard deviations
# simulate_random_params <- function(coef_estimates, sd_estimates, n_sim = 1000) {
#     valid_params <- names(sd_estimates)[!is.na(sd_estimates) & sd_estimates > 0]  # Only use valid standard deviations
#     random_params <- sapply(valid_params, function(param) {
#         rnorm(n_sim, mean = coef_estimates[param], sd = sd_estimates[param])
#     })
#     return(as.data.frame(random_params))
# }
# 
# # Extract the estimated means (coefficients) and standard deviations
# coef_estimates <- coef(M1)
# sd_estimates <- coef(M1)[grepl("^sd\\.", names(coef(M1)))]
# 
# # Ensure the names match the parameter names (remove 'sd.' prefix)
# names(sd_estimates) <- sub("^sd\\.", "", names(sd_estimates))
# 
# # Simulate the random parameters
# random_params_sim <- simulate_random_params(coef_estimates, sd_estimates)
# 
# # Plot histograms of simulated random parameters
# par(mfrow = c(2, 5))  # Adjust the layout as needed
# 
# for (param in names(random_params_sim)) {
#     hist(random_params_sim[[param]], main = param, xlab = "Value", ylab = "Frequency", breaks = 30)
# }







