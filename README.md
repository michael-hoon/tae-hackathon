# The Analytics Edge Data Competition 2024
Repository for 40.016 The Analytics Edge Internal Hackathon

# Directories
- **01 data**: contains the datasets required for the competition.
- **02 eda**: contains our efforts to understand the dataset through exploratory data analysis (EDA).
- **03 models**: contains all models developed by our team.
- **04 submissions**: contains all code used to create Kaggle submissions.
- **05 final submission**: contains our final submission used to evaluate the private leaderboard.
  - `ensemble_finalsubmission.R`: our final ensemble model which is used as the final submission.   
  - `FinalSubmission.csv`: predictions used in the final submission.
  - `train2024_XGBpredictions.csv`, `train2024_MNLpredictions.csv`, `train2024_RFpredictions.csv`: predictions for the `train2024.csv` dataset made by our XGBoost, MNL, and RF models. These predictions are fed into the ensemble model to train and fine-tune the ensemble model.
  - `test2024_XGBpredictions.csv`, `test2024_MNLpredictions.csv`, `test2024_RFpredictions.csv`: predictions for the `test2024.csv` dataset made by our XGBoost, MNL, and RF models. These predictions are fed into the ensemble model to create the final predictions.  
  - `ensemble_analysis.csv`: results from ensemble weights grid-search. 
- **renv**: virtual environment to install all packages in R.  

# Contributors
Michael Hoon \
Nathan Ansel \
Evan Ang Jun Ting \
Wong Qi Yuan Kenneth
