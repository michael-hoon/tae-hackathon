rm(list=ls())
# Execute our custom script for loading packages
source("usePackages.R")
# Name of the packages 
pkgnames <- c("dplyr")
# Use our custom load function
loadPkgs(pkgnames)

# import data
safety <- read.csv("train2024.csv")
str(safety)
head(safety)

# choice variables
safety$Choice <- ifelse(safety$Ch1 == 1, 1,
                        ifelse(safety$Ch2 == 1, 2,
                               ifelse(safety$Ch3 == 1, 3,
                                      ifelse(safety$Ch4 == 1, 4, NA))))
table(safety$Choice)

# drop columns with suspected multicollinearity
# might have collinearity with miles, milesind and milesa
# might have collinearity with night, nightind and nighta
df <- subset(safety, select = -c(Ch1, Ch2, Ch3, Ch4)) #, Case, No, segment,year,miles,night,ppark,gender,age,educ,region,Urb,income
head(df)

# export preprocessed train file
write.csv(df, file = "./train2024_preprocessed.csv", row.names = FALSE)

# repeat for test data
safety <- read.csv("test2024.csv")
str(safety)
head(safety)

# add choice
safety$Choice <- 0
safety$Ch1 <- 0
safety$Ch2 <- 0
safety$Ch3 <- 0
safety$Ch4 <- 0

# drop columns with suspected multicollinearity
# might have collinearity with miles, milesind and milesa
# might have collinearity with night, nightind and nighta
df <- subset(safety, select = -c(Ch1, Ch2, Ch3, Ch4))
head(df)
#, Case, No, segment,year,miles,night,ppark,gender,age,educ,region,Urb,income

# export preprocessed test file
write.csv(df, file = "test2024_preprocessed.csv", row.names = FALSE)