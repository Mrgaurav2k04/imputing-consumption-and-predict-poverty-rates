# imputing-consumption-and-predict-poverty-rates
Using Machine Learning imputation methods, this project predicts household-level daily per capita consumption and estimates population-level poverty rates across various thresholds.
# Poverty Rate Imputation & Consumption Prediction

## 📌 Project Overview
This project focuses on **Poverty Imputation**, a critical task for real-time poverty monitoring. [cite_start]It addresses the challenge where recent surveys lack detailed household consumption data, requiring the use of older, more detailed surveys to "impute" or infer poverty rates and consumption levels[cite: 3, 4].

The goal is to predict both:
1.  **Household-level daily per capita consumption** (in 2017 USD PPP).
2.  **Population-level poverty rates** across 19 different thresholds.

## 📊 The Challenge
The evaluation for this project is highly specialized, reflecting the priorities of the World Bank Group. [cite_start]The performance metric is a **90/10 split**:
* **90% of the score:** Weighted Mean Absolute Percentage Error (WMAPE) of predicted poverty rates.
* **10% of the score:** Mean Absolute Percentage Error (MAPE) of household-level consumption.

The thresholds used for prediction are derived from the ventiles of the consumption distribution of a specific survey (ID 300000).

## 📁 Dataset Structure
The training set consists of three survey panels (IDs 100000, 200000, 300000), each with approximately 35,000 responses.
Features include:
* **Identifiers & Sampling:** Weights used to convert household data to population-level estimates.
* **Socio-economics:** Education, employment, housing, and utilities.
* **Demographics:** Household composition and age.
* **Consumption Indicators:** Food-consumption indicators from the last 7 days.



## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost/CatBoost
* **Imputation Strategy:** Regression-based imputation to infer consumption labels for the test set (Surveys 400000, 500000, 600000).
* **Weighting:** Implementation of population-expanded weights to ensure household predictions accurately reflect the broader population.

## 🚀 How to Run
1. **Clone the Repo:**
   ```bash
   git clone (https://github.com/Mrgaurav2k04/imputing-consumption-and-predict-poverty-rates)
2. **Streamlit Dashboard**
   https://imputing-consumption-and-predict-poverty-rates.streamlit.app/
   
