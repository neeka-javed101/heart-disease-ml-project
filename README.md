# Heart Disease Analysis and Prediction

## Project Overview
Heart disease is one of the leading causes of mortality worldwide. Early detection is critical for effective treatment and management. This project leverages **data-driven methods** to predict the presence of heart disease using clinical and diagnostic features. The project uses the **Heart Disease UCI dataset** and implements an end-to-end workflow including data loading, exploration, preprocessing, EDA, visualization, machine learning modeling, and evaluation of model performance.

## Dataset
- Source: [UCI Machine Learning Repository](kaggle)
- File: `heart_disease_uci.csv`
- Rows: 920
- Columns: 21 features including age, sex, blood pressure, cholesterol, max heart rate, and diagnostic test results.
- Target variable: `num` (0 = no disease, 1 = disease)
- ## Steps and Methodology
1. Load the Data: Imported dataset using `pandas`, explored data types, missing values, and basic statistics.  
2. Exploratory Data Analysis (EDA): Visualized target variable distribution, age, sex, max heart rate, cholesterol, ST depression; generated correlation heatmaps and pairplots.  
3. Data Preprocessing: Handled missing values, encoded categorical variables, scaled numeric features, removed duplicates and unnecessary columns.  
4. Machine Learning Modeling: Applied **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers. Evaluated models using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC Curve. Random Forest achieved highest accuracy (~90%).  
5. Feature Importance: Key predictors: `thalch` (max heart rate), `oldpeak` (ST depression), `age`, `chol`.
