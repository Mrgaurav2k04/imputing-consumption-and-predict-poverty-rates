import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# --- 1. SETTING UP THE WEB PAGE ---
st.set_page_config(page_title="Poverty Imputation Dashboard", page_icon="🌍", layout="wide")

st.title("🌍 Poverty Monitoring & AI Prediction Dashboard")
st.markdown("""
Welcome to the interactive **Poverty Imputation Dashboard**. 
This tool visualizes the output of our Machine Learning models, showing both **Household Consumption** predictions 
and the corresponding extrapolated **Population Poverty Rates** across multiple geographical survey areas.
""")

# --- 2. DATA LOADING SECTION ---
@st.cache_data
def load_predictions():
    try:
        preds_hh = pd.read_csv('predicted_household_consumption.csv')
        preds_pov = pd.read_csv('predicted_poverty_distribution.csv')
        return preds_hh, preds_pov
    except FileNotFoundError:
        st.error("Error: Could not find the prediction CSV files. Please run `model_pipeline.py` first.")
        return None, None

@st.cache_data
def load_training_data():
    try:
        train_feat = pd.read_csv('train_hh_features.csv')
        train_gt = pd.read_csv('train_hh_gt.csv')
        df = pd.merge(train_feat, train_gt, on=['survey_id', 'hhid'])
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_model_metrics():
    try:
        with open('model_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

preds_hh, preds_pov = load_predictions()
train_data = load_training_data()
model_metrics = load_model_metrics()

if preds_hh is not None and preds_pov is not None:
    st.sidebar.success("✅ Prediction Data Loaded")
    if model_metrics:
        st.sidebar.success("✅ Model Metrics Loaded")
    else:
        st.sidebar.warning("⚠️ No model metrics found. Re-run `model_pipeline.py` to generate them.")
    
    # --- 3. DASHBOARD TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Overview", 
        "🎯 Model Performance",
        "📉 Poverty Rate Explorer", 
        "🏠 Household Consumption"
    ])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("Portfolio Overview")
        st.write("Summary statistics based on the unseen test population (surveys 400000, 500000, 600000).")
        
        avg_consumption = preds_hh['cons_ppp17'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Test Households", f"{len(preds_hh):,}")
        col2.metric("Avg Predicted Daily Consumption", f"${avg_consumption:.2f}")
        
        if 'pct_hh_below_3.17' in preds_pov.columns:
            avg_pov = preds_pov['pct_hh_below_3.17'].mean()
            col3.metric("Avg Poverty Rate (at $3.17)", f"{avg_pov:.2%}")
        
        # Show R² in the overview if metrics are available
        if model_metrics:
            r2_log = model_metrics['overall'].get('r2_score_log', model_metrics['overall']['r2_score'])
            col4.metric("Model Accuracy (R²)", f"{r2_log:.4f}")
            
        st.divider()
        st.subheader("Poverty Rate Distribution Matrix")
        st.write("Predicted poverty rates mapping unseen Surveys against all 19 defined thresholds:")
        
        styled_table = preds_pov.style.format({
            col: "{:.1%}" for col in preds_pov.columns if col != 'survey_id'
        })
        st.dataframe(styled_table)

    # --- TAB 2: MODEL PERFORMANCE ---
    with tab2:
        st.header("🎯 Model Performance & Accuracy Metrics")
        st.write("Evaluation results from **3-Fold GroupKFold** Cross-Validation on the training data using **XGBRegressor**.")
        
        if model_metrics:
            overall = model_metrics['overall']
            
            # --- Overall Metrics Cards ---
            st.subheader("Overall Out-of-Fold Performance")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            r2_val = overall['r2_score']
            r2_log_val = overall.get('r2_score_log', r2_val)
            rmse_val = overall['rmse']
            mae_val = overall['mae']
            
            m1.metric(
                label="R² Score (Log-Scale)",
                value=f"{r2_log_val:.4f}",
                help="R² on log-transformed consumption. Primary accuracy metric for skewed economic data. 1.0 = perfect."
            )
            m2.metric(
                label="R² Score (Raw-Scale)",
                value=f"{r2_val:.4f}",
                help="R² on original consumption values. Lower due to extreme outliers in the consumption distribution."
            )
            m3.metric(
                label="RMSE",
                value=f"{rmse_val:.4f}",
                help="Root Mean Squared Error — lower is better. Measures average prediction error magnitude."
            )
            m4.metric(
                label="MAE",
                value=f"{mae_val:.4f}",
                help="Mean Absolute Error — average absolute difference between predicted and actual values."
            )
            m5.metric(
                label="Training Samples",
                value=f"{overall['num_training_samples']:,}",
                help="Total number of household records used for training."
            )
            
            # --- R² Interpretation (based on log-space R²) ---
            st.divider()
            st.subheader("Model Accuracy Interpretation")
            
            if r2_log_val >= 0.9:
                accuracy_color = "🟢"
                accuracy_label = "Excellent"
                accuracy_desc = "The model explains over 90% of variance — highly accurate predictions."
            elif r2_log_val >= 0.7:
                accuracy_color = "🟡"
                accuracy_label = "Good"
                accuracy_desc = "The model captures the majority of patterns in household consumption."
            elif r2_log_val >= 0.5:
                accuracy_color = "🟠"
                accuracy_label = "Moderate"
                accuracy_desc = "The model captures some patterns but there is significant unexplained variance."
            else:
                accuracy_color = "🔴"
                accuracy_label = "Needs Improvement"
                accuracy_desc = "The model struggles to predict consumption accurately."
            
            st.info(f"{accuracy_color} **Model Rating: {accuracy_label}** — {accuracy_desc}")
            
            st.caption(
                "💡 **Why Log-Scale R²?** Household consumption data is highly right-skewed (skew ≈ 3.7). "
                "A few high-consumption outliers disproportionately inflate raw-scale errors. "
                "Log-transformation normalizes the distribution, giving a more representative accuracy measure. "
                "The log-scale R² is the standard metric for economic consumption modeling."
            )
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"""
                | Metric | Value | Meaning |
                |--------|-------|---------|
                | **R² (Log-Scale)** | `{r2_log_val:.4f}` | {r2_log_val*100:.2f}% of log-variance explained |
                | **R² (Raw-Scale)** | `{r2_val:.4f}` | {r2_val*100:.2f}% of raw variance explained |
                | **RMSE** | `{rmse_val:.4f}` | Avg error: ±${rmse_val:.2f}/day |
                | **MAE** | `{mae_val:.4f}` | Avg absolute error: ${mae_val:.2f}/day |
                """)
            with col_info2:
                st.markdown(f"""
                | Config | Value |
                |--------|-------|
                | **Algorithm** | XGBRegressor |
                | **CV Strategy** | 3-Fold GroupKFold |
                | **Features Used** | {overall['num_features']} |
                | **Training Samples** | {overall['num_training_samples']:,} |
                | **Target Transform** | log1p (skew reduction) |
                """)
            
            # --- Per-Fold Breakdown ---
            st.divider()
            st.subheader("Per-Fold Cross-Validation Breakdown")
            st.write("Each fold holds out one entire survey to test out-of-survey generalization:")
            
            fold_df = pd.DataFrame(model_metrics['fold_metrics'])
            if 'r2_score_log' in fold_df.columns:
                fold_df.columns = ['Fold', 'RMSE', 'R² (Raw)', 'R² (Log)', 'MAE']
            else:
                fold_df.columns = ['Fold', 'RMSE', 'R² (Raw)', 'MAE']
            
            # Display fold table
            format_dict = {'RMSE': '{:.4f}', 'R² (Raw)': '{:.4f}', 'MAE': '{:.4f}'}
            highlight_r2_col = 'R² (Raw)'
            if 'R² (Log)' in fold_df.columns:
                format_dict['R² (Log)'] = '{:.4f}'
                highlight_r2_col = 'R² (Log)'
            
            st.dataframe(
                fold_df.style.format(format_dict)
                .highlight_min(subset=['RMSE', 'MAE'], color='#d4edda')
                .highlight_max(subset=[highlight_r2_col], color='#d4edda'),
                use_container_width=True,
                hide_index=True
            )
            
            # Bar chart of fold metrics
            st.write("")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.write("**RMSE by Fold**")
                rmse_chart = pd.DataFrame({
                    'RMSE': fold_df['RMSE'].values
                }, index=[f"Fold {i+1}" for i in range(len(fold_df))])
                st.bar_chart(rmse_chart)
            
            with chart_col2:
                r2_col = 'R² (Log)' if 'R² (Log)' in fold_df.columns else 'R² (Raw)'
                st.write(f"**{r2_col} by Fold**")
                r2_chart = pd.DataFrame({
                    r2_col: fold_df[r2_col].values
                }, index=[f"Fold {i+1}" for i in range(len(fold_df))])
                st.bar_chart(r2_chart)
                
        else:
            st.warning(
                "⚠️ **Model metrics not found.** Please re-run `model_pipeline.py` to generate "
                "`model_metrics.json` with accuracy and RMSE data."
            )
            st.code("python model_pipeline.py", language="bash")

    # --- TAB 3: POVERTY EXPLORER ---
    with tab3:
        st.header("📉 Poverty Rate Extrapolation Curves")
        st.write("Tracing the predicted poverty concentration over increasing expenditure thresholds by survey.")
        
        if not preds_pov.empty:
            pov_melt = preds_pov.melt(id_vars='survey_id', var_name='threshold', value_name='poverty_rate')
            pov_melt['threshold_value'] = pov_melt['threshold'].str.replace('pct_hh_below_', '').astype(float)
            chart_data = pov_melt.pivot(index='threshold_value', columns='survey_id', values='poverty_rate')
            
            st.line_chart(chart_data)
            st.caption("X-Axis: Daily Consumption Threshold ($ PPP17) | Y-Axis: Predicted % of Population in Poverty")

    # --- TAB 4: HOUSEHOLD CONSUMPTION ---
    with tab4:
        st.header("🏠 Micro-Economic Household Distributions")
        st.write("Analyzing the raw spread of predicted daily per capita consumptions across individual households.")
        
        survey_filter = st.selectbox("Select Survey Area to Inspect:", ["All"] + list(preds_hh['survey_id'].unique()))
        
        if survey_filter != "All":
            filtered_hh = preds_hh[preds_hh['survey_id'] == survey_filter]
        else:
            filtered_hh = preds_hh
            
        bins = np.linspace(0, filtered_hh['cons_ppp17'].max(), 50)
        hist_values, _ = np.histogram(filtered_hh['cons_ppp17'], bins=bins)
        
        chart_dict = pd.DataFrame({"Count of Households": hist_values}, index=np.round(bins[:-1], 2))
        
        st.bar_chart(chart_dict)
        st.caption(f"Consumption Density Histogram for Survey Area: {survey_filter}")
        
else:
    st.info("Please generate the predictions to view the dashboard.")