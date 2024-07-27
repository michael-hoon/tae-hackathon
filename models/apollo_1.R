setwd("C:\\SUTD\\40.016 TAE - The Analytics Edge\\000 - COMPETITION\\models")
library(apollo)
library(dplyr)

safety <- read.csv("train2024.csv")

# DATA SPLITTING
safety$Choice <- ifelse(safety$Ch1 == 1, 1, 
                        ifelse(safety$Ch2 == 1, 2, 
                               ifelse(safety$Ch3 == 1, 3, 4)))
train_raw <- subset(safety, Task <= 12)
test_raw <- subset(safety, Task > 12)

# Prepare Apollo data format
train <- subset(train_raw, select=-c(Task,Ch1,Ch2,Ch3,Ch4,educ,gender,region,segment,
                                     ppark,night,miles,Case,No,CC4,GN4,NS4,BU4,FA4,
                                     LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,
                                     LB4,AF4,HU4,Price4,Urb,income,age))

# Initialize Apollo
apollo_initialise()

# Set core controls
apollo_control = list(
  modelName       = "DCM_safety",
  modelDescr      = "Discrete choice model on safety data",
  indivID         = "ID",  
  nCores          = 4,
  outputDirectory = "output"
)

# Define model parameters
apollo_beta = c(
  b_CC1 = 0,
  b_CC2 = 0,
  b_CC3 = 0,
  b_CC4 = 0
)

apollo_fixed = c()

# Load and transform data
database <- train
database$choice <- as.integer(database$Choice)
database$price <- database$Price1 # Assuming Price1 is one of the variables

# Define settings for the choice model
choiceAnalysis_settings <- list(
  alternatives = c(alt1=1, alt2=2, alt3=3, alt4=4),
  choiceVar    = database$choice,
  explanators  = database[, c("CC1", "CC2", "CC3")]
)

# Define likelihood function
apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  P = list()
  
  V = list()
  V[["alt1"]] = b_price * database$Price1 + b_safety * database$Safety1 + b_comfort * database$Comfort1
  V[["alt2"]] = b_price * database$Price2 + b_safety * database$Safety2 + b_comfort * database$Comfort2
  V[["alt3"]] = b_price * database$Price3 + b_safety * database$Safety3 + b_comfort * database$Comfort3
  V[["alt4"]] = b_price * database$Price4 + b_safety * database$Safety4 + b_comfort * database$Comfort4
  
  mnl_settings = list(
    alternatives  = c(alt1=1, alt2=2, alt3=3, alt4=4),
    avail         = list(alt1=1, alt2=1, alt3=1, alt4=1),
    choiceVar     = database$choice,
    utilities     = V
  )
  
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  P = apollo_panelProd(P, apollo_inputs, functionality)
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# Validate inputs
apollo_inputs = apollo_validateInputs()

# Model estimation
model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

# Model outputs
apollo_modelOutput(model)
apollo_saveOutput(model)

# Test the model on the test set
database_test <- test_raw
database_test$choice <- as.integer(database_test$Choice)
database_test$price <- database_test$Price1

# Redefine likelihood function for test set
apollo_probabilities_test=function(apollo_beta, apollo_inputs, functionality="validate"){
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  P = list()
  
  V = list()
  V[["alt1"]] = b_price * database_test$Price1 + b_safety * database_test$Safety1 + b_comfort * database_test$Comfort1
  V[["alt2"]] = b_price * database_test$Price2 + b_safety * database_test$Safety2 + b_comfort * database_test$Comfort2
  V[["alt3"]] = b_price * database_test$Price3 + b_safety * database_test$Safety3 + b_comfort * database_test$Comfort3
  V[["alt4"]] = b_price * database_test$Price4 + b_safety * database_test$Safety4 + b_comfort * database_test$Comfort4
  
  mnl_settings = list(
    alternatives  = c(alt1=1, alt2=2, alt3=3, alt4=4),
    avail         = list(alt1=1, alt2=1, alt3=1, alt4=1),
    choiceVar     = database_test$choice,
    utilities     = V
  )
  
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  P = apollo_panelProd(P, apollo_inputs, functionality)
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# Validate inputs for test set
apollo_inputs_test = apollo_validateInputs()

# Predict on the test set
test_probabilities = apollo_probabilities_test(model$estimate, apollo_inputs_test, functionality="validate")
test_probabilities
