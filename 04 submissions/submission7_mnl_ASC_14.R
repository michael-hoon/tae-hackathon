
library(mlogit)

# DATA EXTRACTION
df_train <- read.csv("train2024.csv")
df_test <- read.csv("test2024.csv")

df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

taskS <- dfidx(subset(df_train, Task<=14), shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))
valS <- dfidx(subset(df_train, Task>14), shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))
test <- dfidx(df_test, shape="wide", choice="Choice", sep="", varying = c(4:83), idx = c("No", "Case"))

# MNL with only significant features (Removed CC, segmentind, educind, nightind)
M_ASC <- mlogit(Choice~GN+NS+BU+FA+LD+BZ+FC+FP+RP+PP+KA+SC+TS+NV+MA+LB+AF+HU+Price-1 | milesind+pparkind+genderind+ageind+Urbind, data=taskS)
summary(M_ASC)
M_ASC$logLik/nrow(train)

# Predicted probabilities of MNL with all safety and human features
P_vl_ASC <- predict(M_ASC, newdata=valS)

val_task_split_probs <- subset(df_train, Task > 14)[110:113]
# Mean log likelihood
LL4 <- 0

for (i in 1:nrow(P_vl_ASC)){
  for (j in 1:ncol(P_vl_ASC)){
    LL4 <- LL4 + val_task_split_probs[i,j]*log(P_vl_ASC[i,j])
  }
}
LL4
LL4/nrow(P_vl_ASC)

MNL_ASC_probs2 <- predict(M_ASC, newdata = test)

#write.csv(MNL_ASC_probs, "/Users/kennethwong/Documents/Term5_TAE/competition/to_submit.csv", row.names=FALSE)