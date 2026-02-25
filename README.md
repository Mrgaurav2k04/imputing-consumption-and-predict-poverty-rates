# imputing-consumption-and-predict-poverty-rates
Using Machine Learning imputation methods, this project predicts household-level daily per capita consumption and estimates population-level poverty rates across various thresholds.
# Poverty Rate Imputation & Consumption Prediction

## 📌 Project Overview
This project focuses on **Poverty Imputation**, a critical task for real-time poverty monitoring. [cite_start]It addresses the challenge where recent surveys lack detailed household consumption data, requiring the use of older, more detailed surveys to "impute" or infer poverty rates and consumption levels[cite: 3, 4].

The goal is to predict both:
1.  [cite_start]**Household-level daily per capita consumption** (in 2017 USD PPP)[cite: 2, 35].
2.  [cite_start]**Population-level poverty rates** across 19 different thresholds[cite: 2, 41].

## 📊 The Challenge
The evaluation for this project is highly specialized, reflecting the priorities of the World Bank Group. [cite_start]The performance metric is a **90/10 split**[cite: 55]:
* [cite_start]**90% of the score:** Weighted Mean Absolute Percentage Error (WMAPE) of predicted poverty rates[cite: 55].
* [cite_start]**10% of the score:** Mean Absolute Percentage Error (MAPE) of household-level consumption[cite: 59].

[cite_start]The thresholds used for prediction are derived from the ventiles of the consumption distribution of a specific survey (ID 300000)[cite: 39, 56].

## 📁 Dataset Structure
[cite_start]The training set consists of three survey panels (IDs 100000, 200000, 300000), each with approximately 35,000 responses[cite: 43, 44].
Features include:
* [cite_start]**Identifiers & Sampling:** Weights used to convert household data to population-level estimates[cite: 24, 25, 26].
* [cite_start]**Demographics:** Household composition and age[cite: 28].
* [cite_start]**Socio-economics:** Education, employment, housing, and utilities[cite: 29, 30].
* [cite_start]**Consumption Indicators:** Food-consumption indicators from the last 7 days[cite: 32].



## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost/CatBoost
* [cite_start]**Imputation Strategy:** Regression-based imputation to infer consumption labels for the test set (Surveys 400000, 500000, 600000)[cite: 46, 47].
* [cite_start]**Weighting:** Implementation of population-expanded weights to ensure household predictions accurately reflect the broader population[cite: 11, 25].

## 🚀 How to Run
1. **Clone the Repo:**
   ```bash
   git clone [https://https://github.com/Mrgaurav2k04/imputing-consumption-and-predict-poverty-rates)
