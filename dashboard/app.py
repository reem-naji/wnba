import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Set page configuration for a wide layout dashboard
st.set_page_config(page_title="2026 WNBA MVP Race Predictions", layout="wide")

# 1. Load Data with caching for better performance
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
@st.cache_data
def load_data():
    # Adjusted path assuming data is relative to your running script
    data = pd.read_csv(PROJECT_ROOT / 'ml' / 'data'/ 'predictions.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Feature engineering from your notebook
    data['mvp_caliber'] = np.where(data['Predicted_Share'] >= 0.1 , 'yes', 'no')
    return data

try:
    data = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'data/predictions.csv' exists.")
    st.stop()

# Dashboard Title
st.title("🏀 2026 WNBA MVP Race Predictions & Analytics")
    

st.markdown("---")

# --- ROW 1: TOP 5 MVP LEADERS & CORRELATION ---
col1, col2 = st.columns([1, 2])

with col2:
    st.subheader("Top 5 MVP Contenders")
    st.markdown("Current leaders based on predicted award share:")

    # Displaying the top 5 players exactly like your second cell
    top_5 = data[['Player', 'Team', 'Pos', 'Predicted_Share']].head(5)
    
    # Quick KPI Metric card for the frontrunner
    # Styled dataframe
    st.dataframe(
        top_5.style.format({'Predicted_Share': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )

    

with col1:
    st.subheader("🏆 MVP Frontrunner")
    leader_name = data.iloc[0]['Player']
    leader_share = data.iloc[0]['Predicted_Share']
    st.metric(label="",value=leader_name, delta=f"{leader_share:.2%} Pred. Share")


st.markdown("---")

# --- ROW 2: INTERACTIVE SCATTER PLOTS (2x4 Grid Translation) ---
st.subheader("📈 Impact of Statistical Features on Win Shares")
st.markdown("Hover over data points to identify individual players and track performance profiles.")

# Recreating your 2x4 seaborn grid using Streamlit layout columns
features = [
    ('pts_per_game', 'Points Per Game'),
    ('PER', 'Player Efficiency Rating (PER)'),
    ('USG%', 'Usage Percentage (USG%)'),
    ('TS%', 'True Shooting % (TS%)'),
    ('ast_per_game', 'Assists Per Game'),
    ('trb_per_game', 'Total Rebounds Per Game'),
    ('blk_per_game', 'Blocks Per Game'),
    ('stl_per_game', 'Steals Per Game')
]

# Grid Configuration: Row 1
row1_cols = st.columns(4)
for i in range(4):
    feat_col, feat_label = features[i]
    with row1_cols[i]:
        fig = px.scatter(
            data, x=feat_col, y='WS',
            color='Predicted_Share',
            color_continuous_scale='plasma',
            hover_name='Player',
            hover_data=['Team', 'Pos', 'Predicted_Share'],
            labels={feat_col: feat_label, 'WS': 'Win Shares (WS)'}
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

# Grid Configuration: Row 2
row2_cols = st.columns(4)
for i in range(4, 8):
    feat_col, feat_label = features[i]
    with row2_cols[i-4]:
        fig = px.scatter(
            data, x=feat_col, y='WS',
            color='Predicted_Share',
            color_continuous_scale='plasma',
            hover_name='Player',
            hover_data=['Team', 'Pos', 'Predicted_Share'],
            labels={feat_col: feat_label, 'WS': 'Win Shares (WS)'}
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col3, col4 = st.columns([1,1])
with col3:
    st.subheader("📊 Feature Correlation Matrix")
    
    # Generate Spearman correlation matrix from your notebook
    corr_matrix = data[data.columns[4:14]].corr(method='spearman')
    
    # Interactive Plotly Heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="plasma",
        aspect="auto",
        labels=dict(color="Spearman Corr")
    )
    fig_corr.update_layout(margin=dict(l=25, r=25, t=25, b=25), height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

with col4:
    st.subheader("🔍 Deep Dive: Interactive Distribution Explorer")
    st.markdown("Select any specific feature to view a highly detailed distribution, including a marginal box plot to easily spot outliers.")

    # Allow the user to select any numeric column using a dropdown
    numeric_columns = ['Predicted_Share', 'WS'] + [f[0] for f in features]
    selected_feat = st.selectbox("Select a metric to explore:", options=numeric_columns, index=0)

    # Create a detailed histogram with a marginal box plot
    fig_detailed_hist = px.histogram(
        data, 
        x=selected_feat, 
        nbins=40, 
        marginal="box", # Adds a box plot above the histogram to show quartiles/outliers
        hover_data=['Player', 'Team'], # Shows player names when hovering over outliers
        color_discrete_sequence=["#477E9F"]
    )
    fig_detailed_hist.update_layout(bargap=0.05, height=400)

    st.plotly_chart(fig_detailed_hist, use_container_width=True)
    # --- ROW 3: FEATURE DISTRIBUTIONS ---
st.markdown("---")
st.subheader("📊 Statistical Feature Distributions")
st.markdown("Explore how the key player statistics are distributed across the league.")

# We will reuse the 'features' list from the scatter plots to create a grid of histograms
# Create a 2x4 grid using Streamlit columns
for i in range(0, len(features), 4):
    hist_cols = st.columns(4)
    for j in range(4):
        if i + j < len(features):
            feat_col, feat_label = features[i + j]
            with hist_cols[j]:
                # Create an interactive histogram with Plotly
                fig_hist = px.histogram(
                    data, 
                    x=feat_col, 
                    nbins=25,
                    color_discrete_sequence=["#477E9F"], # A clean, professional blue
                    labels={feat_col: feat_label}
                )
                
                # Update layout for a compact grid look
                fig_hist.update_layout(
                    title=dict(text=feat_col, font=dict(size=14)),
                    margin=dict(l=10, r=10, t=40, b=10), 
                    height=250,
                    bargap=0.1,
                    yaxis_title="Count" if j == 0 else "" # Only show y-axis label on the far left
                )
                
                # Render the chart
                st.plotly_chart(fig_hist, use_container_width=True)

# --- OPTIONAL: SINGLE DETAILED DISTRIBUTION EXPLORER ---
st.markdown("---")
