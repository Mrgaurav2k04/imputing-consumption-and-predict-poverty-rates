# 📊 Poverty Rate Imputation & Prediction Model

Using Machine Learning imputation methods, this project predicts household-level daily per capita consumption and estimates population-level poverty rates across various thresholds. It features a state-of-the-art interactive executive dashboard for macro-economic analysis.

## 🌟 Executive Dashboard Features
The project includes a highly professional, interactive web dashboard built with Streamlit and Plotly to visualize the machine learning outputs:
- **Advanced Infographics:** Features interactive Plotly Treemaps (for regional poverty concentration), Donut charts (for aggregate thresholds), and Violin plots (for household consumption density).
- **Professional UI/UX:** Built with a sleek dark-slate theme, glassmorphism metric cards, and a continuously moving high-tech data grid background.
- **Live Data Simulation:** A toggleable live-feed mode that animates the extrapolation of poverty rate curves point-by-point.
- **Model Analytics:** Real-time gauge charts displaying the ensemble's R² and RMSE scores, along with a detailed breakdown of Cross-Validation folds.

## 📌 Project Overview
This project focuses on Poverty Imputation, a critical task for real-time poverty monitoring. It addresses the challenge where recent surveys lack detailed household consumption data, requiring the use of older, more detailed surveys to "impute" or infer poverty rates and consumption levels.

The goal is to predict both:
1. Household-level daily per capita consumption (in 2017 USD PPP).
2. Population-level poverty rates across 19 different thresholds.

## 📊 The Challenge
The evaluation for this project is highly specialized, reflecting the priorities of the World Bank Group. The performance metric is a 90/10 split:
- **90% of the score:** Weighted Mean Absolute Percentage Error (WMAPE) of predicted poverty rates.
- **10% of the score:** Mean Absolute Percentage Error (MAPE) of household-level consumption.

The thresholds used for prediction are derived from the ventiles of the consumption distribution of a specific survey (ID 300000).

## 📁 Dataset Structure
The training set consists of three survey panels (IDs 100000, 200000, 300000), each with approximately 35,000 responses. Features include:
- **Identifiers & Sampling:** Weights used to convert household data to population-level estimates.
- **Socio-economics:** Education, employment, housing, and utilities.
- **Demographics:** Household composition and age.
- **Consumption Indicators:** Food-consumption indicators from the last 7 days.

## 🛠️ Tech Stack & Methodology
### Machine Learning Backend
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost
- **Imputation Strategy:** Advanced ensemble regression to infer consumption labels for the test set.
- **Weighting:** Implementation of population-expanded weights to ensure household predictions accurately reflect the broader population.

### Frontend Dashboard
- **Framework:** Streamlit
- **Visualization:** Plotly (Express & Graph Objects)
- **Animations:** Streamlit-Lottie & Custom CSS Keyframes


## 🚀 How to Run
1. **Clone the Repo:**
   ```bash
   git clone (https://github.com/Mrgaurav2k04/imputing-consumption-and-predict-poverty-rates)
2. **Streamlit Dashboard**
   https://imputing-consumption-and-predict-poverty-rates.streamlit.app/
   
