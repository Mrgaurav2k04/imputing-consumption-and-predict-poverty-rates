import streamlit as st
import pandas as pd
import numpy as np

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
        # Load the generated ML outputs
        preds_hh = pd.read_csv('predicted_household_consumption.csv')
        preds_pov = pd.read_csv('predicted_poverty_distribution.csv')
        return preds_hh, preds_pov
    except FileNotFoundError:
        st.error("Error: Could not find the prediction CSV files. Please run `model_pipeline.py` first.")
        return None, None

@st.cache_data
def load_training_data():
    try:
        # Load underlying training datasets for EDA
        train_feat = pd.read_csv('train_hh_features.csv')
        train_gt = pd.read_csv('train_hh_gt.csv')
        df = pd.merge(train_feat, train_gt, on=['survey_id', 'hhid'])
        return df
    except FileNotFoundError:
        return None

preds_hh, preds_pov = load_predictions()
train_data = load_training_data()

if preds_hh is not None and preds_pov is not None:
    st.sidebar.success("✅ Prediction Data Loaded")
    
    # --- 3. DASHBOARD TABS ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Portfolio Overview", 
        "📉 Poverty Rate Explorer", 
        "🏠 Household Consumption"
    ])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("Portfolio Overview")
        st.write("Summary statistics based on the unseen test population (surveys 400000, 500000, 600000).")
        
        # High Level Metrics
        avg_consumption = preds_hh['cons_ppp17'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Test Households", f"{len(preds_hh):,}")
        col2.metric("Average Predicted Daily Consumption", f"${avg_consumption:.2f}")
        
        # We can dynamically pull poverty at a specific threshold (e.g., $3.17)
        if 'pct_hh_below_3.17' in preds_pov.columns:
            avg_pov = preds_pov['pct_hh_below_3.17'].mean()
            col3.metric("Avg Poverty Rate (at $3.17)", f"{avg_pov:.2%}")
            
        st.divider()
        st.subheader("Poverty Rate Distribution Matrix")
        st.write("Predicted poverty rates mapping unseen Surveys against all 19 defined thresholds:")
        
        # Display styled table
        styled_table = preds_pov.style.format({
            col: "{:.1%}" for col in preds_pov.columns if col != 'survey_id'
        })
        
        st.dataframe(styled_table)


    # --- TAB 2: POVERTY EXPLORER ---
    with tab2:
        st.header("📉 Poverty Rate Extrapolation Curves")
        st.write("Tracing the predicted poverty concentration over increasing expenditure thresholds by survey.")
        
        if not preds_pov.empty:
            # Transform data for plotting
            pov_melt = preds_pov.melt(id_vars='survey_id', var_name='threshold', value_name='poverty_rate')
            
            # Clean threshold strings "pct_hh_below_3.17" -> 3.17
            pov_melt['threshold_value'] = pov_melt['threshold'].str.replace('pct_hh_below_', '').astype(float)
            
            # Create a line chart by survey ID using st.line_chart
            # Pivot table to make columns = survey_id, row = thresholds
            chart_data = pov_melt.pivot(index='threshold_value', columns='survey_id', values='poverty_rate')
            
            st.line_chart(chart_data)
            st.caption("X-Axis: Daily Consumption Threshold ($ PPP17) | Y-Axis: Predicted % of Population in Poverty")

    # --- TAB 3: HOUSEHOLD CONSUMPTION ---
    with tab3:
        st.header("🏠 Micro-Economic Household Distributions")
        st.write("Analyzing the raw spread of predicted daily per capita consumptions across individual households.")
        
        # Survey filtering
        survey_filter = st.selectbox("Select Survey Area to Inspect:", ["All"] + list(preds_hh['survey_id'].unique()))
        
        if survey_filter != "All":
            filtered_hh = preds_hh[preds_hh['survey_id'] == survey_filter]
        else:
            filtered_hh = preds_hh
            
        # Draw a histogram using numpy & bar_chart natively
        bins = np.linspace(0, filtered_hh['cons_ppp17'].max(), 50)
        hist_values, _ = np.histogram(filtered_hh['cons_ppp17'], bins=bins)
        
        chart_dict = pd.DataFrame({"Count of Households": hist_values}, index=np.round(bins[:-1], 2))
        
        st.bar_chart(chart_dict)
        st.caption(f"Consumption Density Histogram for Survey Area: {survey_filter}")
        
else:
    st.info("Please generate the predictions to view the dashboard.")