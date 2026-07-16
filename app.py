import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# --- 1. SETTING UP THE WEB PAGE ---
st.set_page_config(page_title="Poverty Imputation & Prediction Model", page_icon="📊", layout="wide")

# --- LOTTIE ANIMATION LOADER ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- CSS INJECTION FOR PROFESSIONAL THEME & LIVE GRID BACKGROUND ---
st.markdown("""
<style>
/* Professional Dark Slate Background */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a; /* Slate 900 */
}

/* Continuous Live Moving Grid Background */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: -40px; left: 0; width: 100vw; height: calc(100vh + 40px);
    background-image: 
        linear-gradient(to right, rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: moveGrid 4s linear infinite;
    z-index: -1;
    pointer-events: none;
}

@keyframes moveGrid {
    0% { transform: translateY(0); }
    100% { transform: translateY(40px); }
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Clean Metric Cards */
div[data-testid="metric-container"] {
    background: #1e293b;
    border-radius: 8px;
    padding: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 15px rgba(59, 130, 246, 0.2);
    border-color: rgba(59, 130, 246, 0.4);
}

h1, h2, h3, p, span, div, label {
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)


col_title, col_anim = st.columns([4, 1])
with col_title:
    st.title("📊 Poverty Rate Imputation and Prediction Model")
    st.markdown("""
    Welcome to the executive dashboard. 
    This application visualizes the output of our ensemble machine learning architecture, extrapolating predicted household consumption into actionable macroeconomic poverty indicators.
    """)
with col_anim:
    # A lightweight data visualization lottie animation
    lottie_header = load_lottieurl("https://lottie.host/801a2f9d-16cb-4029-9c3f-c305aeb2d8b4/YnQ3n4rR9G.json")
    if lottie_header:
        st_lottie(lottie_header, height=120, key="header_anim")


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
def load_model_metrics():
    try:
        with open('model_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

preds_hh, preds_pov = load_predictions()
model_metrics = load_model_metrics()

# --- SIDEBAR & LIVE SIMULATION TOGGLE ---
st.sidebar.title("Controls & Settings")
lottie_sidebar = load_lottieurl("https://lottie.host/960b2169-231a-4c2f-b44c-0062a74c43ba/rFp017H4Nn.json")
if lottie_sidebar:
    with st.sidebar:
        st_lottie(lottie_sidebar, height=150, key="sidebar_anim")

live_simulation = st.sidebar.toggle("🔴 Simulate Live Data Feed", value=False, help="Animate the charts as if data is streaming in real-time.")

if preds_hh is not None and preds_pov is not None:
    st.sidebar.success("✅ Model Data Synchronized")
    
    # --- 3. DASHBOARD TABS ---
    tab1, tab2, tab3 = st.tabs([
        "📈 Demographics & Infographics", 
        "🎯 Model Analytics",
        "📉 Poverty Rate Explorer"
    ])
    
    # --- TAB 1: OVERVIEW & INFOGRAPHICS ---
    with tab1:
        st.header("Executive Summary & Demographics")
        st.write("Visualizing the predicted poverty landscape across the surveyed regions.")
        
        avg_consumption = preds_hh['cons_ppp17'].mean()
        avg_pov = preds_pov['pct_hh_below_3.17'].mean() if 'pct_hh_below_3.17' in preds_pov.columns else 0
        
        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Households Scored", f"{len(preds_hh):,}")
        col2.metric("Mean Daily Consumption", f"${avg_consumption:.2f}")
        col3.metric("Aggregate Poverty Rate ($3.17)", f"{avg_pov:.1%}")
        if model_metrics:
            r2_val = model_metrics['overall'].get('r2_score_log', model_metrics['overall']['r2_score'])
            col4.metric("AI Confidence (R²)", f"{r2_val:.2f}")
            
        st.divider()
        
        # High-End Infographics Layout
        info_col1, info_col2 = st.columns([3, 2])
        
        with info_col1:
            st.subheader("Regional Poverty Distribution (Treemap)")
            st.write("Size represents population count; color represents the severity of poverty.")
            
            # Prepare data for treemap
            if 'pct_hh_below_3.17' in preds_pov.columns:
                tree_df = preds_pov[['survey_id', 'pct_hh_below_3.17']].copy()
                tree_df['survey_id'] = tree_df['survey_id'].astype(str)
                # Count households per survey
                hh_counts = preds_hh.groupby('survey_id').size().reset_index(name='household_count')
                hh_counts['survey_id'] = hh_counts['survey_id'].astype(str)
                tree_df = pd.merge(tree_df, hh_counts, on='survey_id')
                
                fig_tree = px.treemap(
                    tree_df,
                    path=[px.Constant("All Regions"), 'survey_id'],
                    values='household_count',
                    color='pct_hh_below_3.17',
                    color_continuous_scale='Teal',
                    hover_data=['pct_hh_below_3.17'],
                    title="Poverty Concentration by Survey Region"
                )
                fig_tree.update_layout(margin=dict(t=50, l=10, r=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_tree, use_container_width=True)
                
        with info_col2:
            st.subheader("Aggregate Status")
            st.write("Proportion of households classified under the $3.17 threshold.")
            
            # Donut Chart for overall poverty
            labels = ['Below Threshold (Poor)', 'Above Threshold (Non-Poor)']
            values = [avg_pov, 1 - avg_pov]
            
            fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
            fig_donut.update_traces(
                hoverinfo='label+percent', 
                textinfo='percent', 
                textfont_size=20,
                marker=dict(colors=['#ef4444', '#10b981'], line=dict(color='#0f172a', width=2))
            )
            fig_donut.update_layout(
                title_text="Poverty Threshold Classification",
                paper_bgcolor='rgba(0,0,0,0)', 
                font_color='white',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        st.divider()
        with st.expander("🔍 Click to view full Distribution Matrix", expanded=False):
            st.write("Predicted poverty rates mapping unseen Surveys against all defined thresholds:")
            styled_table = preds_pov.style.format({
                col: "{:.1%}" for col in preds_pov.columns if col != 'survey_id'
            }).background_gradient(cmap='Blues', axis=1)
            st.dataframe(styled_table, use_container_width=True)

    # --- TAB 2: MODEL PERFORMANCE (GAUGE CHARTS) ---
    with tab2:
        st.header("🎯 Predictive Model Analytics")
        st.write("Diagnostic evaluation of the Ensemble Machine Learning architecture.")
        
        if model_metrics:
            overall = model_metrics['overall']
            r2_val = overall['r2_score']
            r2_log_val = overall.get('r2_score_log', r2_val)
            rmse_val = overall['rmse']
            
            # --- Gauge Charts ---
            st.subheader("Overall Predictive Strength")
            g_col1, g_col2 = st.columns(2)
            
            with g_col1:
                fig_gauge_r2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=r2_log_val,
                    title={'text': "R² Accuracy Score (Log Scale)"},
                    gauge={'axis': {'range': [0, 1]},
                           'bar': {'color': "#3b82f6"},
                           'steps': [
                               {'range': [0, 0.5], 'color': "rgba(239, 68, 68, 0.2)"},
                               {'range': [0.5, 0.8], 'color': "rgba(234, 179, 8, 0.2)"},
                               {'range': [0.8, 1.0], 'color': "rgba(34, 197, 94, 0.2)"}
                           ]}
                ))
                fig_gauge_r2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
                st.plotly_chart(fig_gauge_r2, use_container_width=True)
                
            with g_col2:
                fig_gauge_rmse = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rmse_val,
                    title={'text': "Root Mean Squared Error (RMSE)"},
                    gauge={'axis': {'range': [0, rmse_val * 2]},
                           'bar': {'color': "#ef4444"},
                           'steps': [
                               {'range': [0, rmse_val], 'color': "rgba(34, 197, 94, 0.2)"},
                               {'range': [rmse_val, rmse_val * 2], 'color': "rgba(239, 68, 68, 0.2)"}
                           ]}
                ))
                fig_gauge_rmse.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
                st.plotly_chart(fig_gauge_rmse, use_container_width=True)

            st.divider()
            
            # --- Violin Plot inside Analytics ---
            st.subheader("Micro-Economic Household Distributions")
            st.write("Advanced Violin/Box plot mapping the probability density of household consumption predictions.")
            
            fig_violin = px.violin(
                preds_hh,
                y="cons_ppp17",
                x="survey_id",
                box=True,
                points="outliers",
                title=f"Consumption Density & Box Plot by Survey",
                labels={"cons_ppp17": "Predicted Daily Consumption ($)", "survey_id": "Survey Region"},
                color="survey_id",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_violin.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.02)',
                font_color='white',
                yaxis_title="Consumption ($ PPP17)",
                showlegend=False
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
            st.divider()
            st.subheader("Cross-Validation Fold Metrics Breakdown")
            
            fold_df = pd.DataFrame(model_metrics['fold_metrics'])
            if 'r2_score_log' in fold_df.columns:
                fold_df.columns = ['Fold', 'RMSE', 'R² (Raw)', 'R² (Log)', 'MAE']
            else:
                fold_df.columns = ['Fold', 'RMSE', 'R² (Raw)', 'MAE']
            
            fold_df['Fold Name'] = [f"Fold {i+1}" for i in range(len(fold_df))]
            
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_rmse = px.bar(fold_df, x='Fold Name', y='RMSE', text='RMSE', title='RMSE by Fold', color='RMSE', color_continuous_scale='Blues')
                fig_rmse.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_rmse.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=50, b=0))
                st.plotly_chart(fig_rmse, use_container_width=True)
            with chart_col2:
                r2_col = 'R² (Log)' if 'R² (Log)' in fold_df.columns else 'R² (Raw)'
                fig_r2 = px.bar(fold_df, x='Fold Name', y=r2_col, text=r2_col, title=f'{r2_col} by Fold', color=r2_col, color_continuous_scale='Teal')
                fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_r2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=50, b=0))
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with st.expander("Detailed Cross-Validation Fold Metrics", expanded=False):
                st.dataframe(fold_df.drop(columns=['Fold Name']), use_container_width=True)

    # --- TAB 3: POVERTY EXPLORER (SPLINE & LIVE SIMULATION) ---
    with tab3:
        st.header("📉 Poverty Rate Extrapolation Curves")
        
        if not preds_pov.empty:
            pov_melt = preds_pov.melt(id_vars='survey_id', var_name='threshold', value_name='poverty_rate')
            pov_melt['threshold_value'] = pov_melt['threshold'].str.replace('pct_hh_below_', '').astype(float)
            pov_melt['survey_id'] = pov_melt['survey_id'].astype(str)
            pov_melt = pov_melt.sort_values(by='threshold_value')
            
            chart_placeholder = st.empty()
            
            if live_simulation:
                # Simulate data arriving point by point
                unique_thresholds = sorted(pov_melt['threshold_value'].unique())
                for i in range(1, len(unique_thresholds) + 1):
                    current_thresholds = unique_thresholds[:i]
                    current_data = pov_melt[pov_melt['threshold_value'].isin(current_thresholds)]
                    
                    fig = px.area(
                        current_data, 
                        x='threshold_value', 
                        y='poverty_rate', 
                        color='survey_id', 
                        line_shape='spline', # Smooth curves
                        title="Live Data Feed: Extrapolated Poverty Curves",
                        labels={"threshold_value": "Daily Consumption Threshold ($)", "poverty_rate": "Predicted % in Poverty"}
                    )
                    fig.update_layout(hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)', font_color='white')
                    fig.update_yaxes(tickformat=".1%", range=[0, 1])
                    fig.update_xaxes(range=[0, max(unique_thresholds)])
                    
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.15)
            else:
                # Static smooth area chart
                fig = px.area(
                    pov_melt, 
                    x='threshold_value', 
                    y='poverty_rate', 
                    color='survey_id', 
                    line_shape='spline', # Smooth curves
                    title="Extrapolated Poverty Curves by Threshold",
                    labels={"threshold_value": "Daily Consumption Threshold ($)", "poverty_rate": "Predicted % in Poverty"}
                )
                fig.update_layout(hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)', font_color='white')
                fig.update_yaxes(tickformat=".1%")
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
else:
    st.info("Please generate the predictions to view the dashboard.")