import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import os

st.set_page_config(page_title="2026 WNBA MVP Race Predictions", layout="wide")

# Load Data with caching for better performance
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / 'ml' / 'data'/ 'predictions.csv'
@st.cache_data
def load_data(mtime):
    data = pd.read_csv(PREDICTIONS_PATH)
    feature_importance = pd.read_csv(PROJECT_ROOT / 'ml' / 'data'/ 'feature_importance.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    if 'Unnamed: 0' in feature_importance.columns:
        feature_importance.drop(columns=['Unnamed: 0'], inplace=True)
    
    data['mvp_caliber'] = np.where(data['Predicted_Share'] >= 0.1 , 'yes', 'no')
    return data, feature_importance

try:
    mtime = os.path.getmtime(PREDICTIONS_PATH)
    data, feature_importance = load_data(mtime)
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'data/predictions.csv' or 'data/feature_importance.csv' exist.")
    st.stop()

st.title("2026 WNBA MVP Race Predictions & Analytics 🏀 ")
    

st.markdown("---")
leader_name = data.iloc[0]['Player']
features = [
    ('pts_per_game', 'Points per game'),
    ('ast_per_game', 'Assists Per Game'),
    ('trb_per_game', 'Total Rebounds Per Game'),
    ('USG%', 'Usage Percentage (USG%)'),
    ('PER', 'Player Efficiency Rating (PER)'),
    ('blk_per_game', 'Blocks Per Game'),
    ('stl_per_game', 'Steals Per Game'),
    ('TS%', 'True Shooting % (TS%)'),
    ('Predicted_Share', 'Predicted Share'),
]

col1, col2, col3 = st.columns([0.5, 1, 1], gap="medium")

with col1:
    
    st.subheader("Predicted MVP 🏆")
    leader_share = data.iloc[0]['Predicted_Share']
    st.metric(label="mvp frontrunner",value=leader_name, delta=f"{leader_share:.2%} Pred. Share", label_visibility="hidden")

with col2:

    st.subheader("Top 5 Predicted MVP Contenders")
    st.markdown("Current leaders based on predicted award share:")

    top_5 = data[['Player', 'Team', 'Pos', 'Predicted_Share']].head(5)
    
    fig_top5 = px.scatter(
            top_5, 
            x='Predicted_Share', 
            y='Player',
            size='Predicted_Share', 
            color_discrete_sequence=['#90D743'],
            hover_data=['Team', 'Pos', 'Predicted_Share']
        )
    
    fig_top5.update_layout(
        margin=dict(l=10, r=20, t=20, b=10),
        height=280,
        xaxis_title="Predicted Share",
        yaxis_title="Player",
    )
    
    st.plotly_chart(fig_top5, use_container_width=True)

with col3:

    st.subheader("Which Stats Matter Most to the Model?")
    st.markdown("The statistics most closely linked to a high predicted award share")

    fig_corr_bar = px.bar(
        feature_importance,
        x='feature',
        y='mean_abs_shap',
        color='mean_abs_shap',              # This creates the continuous color gradient!
        color_continuous_scale='viridis', 
        labels={'mean_abs_shap': 'Feature Importance (Shap)'}
    )
    
    fig_corr_bar.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), 
        height=350,
        xaxis=dict(tickangle=-45, title=None), 
        coloraxis_colorbar=dict(
                orientation="h",  
                yanchor="bottom",
                y=-1,            
                xanchor="center",
                x=0.5,
                thickness=5,       
                title=dict(
                text="Feature Importance",
                side="top",
            )          
            )   
    )
    
    st.plotly_chart(fig_corr_bar, use_container_width=True)

st.markdown("---")

col4, col5 = st.columns([1,1], gap="xxlarge")

with col4:

    st.subheader(f"Where {leader_name} Stands Among the League")
    st.markdown("Comparing players by win shares and points per game — hover for player details, or switch the Y-axis to explore other stats")

    numeric_columns = [f[1] for f in features]
    selected_feat_label = st.selectbox("Select a metric to explore:", options=numeric_columns, index=0, key='metric_filter')
    selected_feat = [f[0] for f in features if f[1] == selected_feat_label][0]

    fig = px.scatter(
        data, y='WS', x=selected_feat,
        color='Predicted_Share', hover_name='Player',
        hover_data=['Team', 'Pos', 'Predicted_Share'],
        color_continuous_scale='viridis',
        labels={'WS': 'Win Shares (WS)',selected_feat: selected_feat_label}
    )
    fig.update_layout(
            margin=dict(l=10, r=10, t=25, b=10), 
            height=400,
            coloraxis_colorbar=dict(
                orientation="h",    # Make colorbar horizontal
                yanchor="bottom",
                y=1.25,            
                xanchor="center",
                x=0.5,
                thickness=5,       
                title=dict(
                text="Predicted Share",
                side="top",
            )          
            )
        )
    st.plotly_chart(fig, use_container_width=True)

number_of_mvp_calibers = len(data['mvp_caliber'] == 'yes')
with col5:
    st.subheader("MVP-Caliber Players vs. the Rest of the League")
    st.markdown(f"How the top {number_of_mvp_calibers} MVP-caliber players — and {leader_name} in particular — separate from the rest of the league")

    features_ = features
    features_.insert(0,('WS', 'Win Shares (WS)'))
    numeric_columns = [f[1] for f in features_]
    selected_feat_label = st.selectbox("Select a metric to explore:", options=numeric_columns, index=0)
    selected_feat = [f[0] for f in features_ if f[1] == selected_feat_label][0]

    # Setup Pills for MULTIPLE selections
    tags = ['MVP Caliber', 'Rest of League']
    
    # selection_mode="multi" allows the user to click both pills. 
    # default=tags makes it so both are selected when the app first loads.
    selected_tags = st.pills(
        'Filter using: ', 
        tags, 
        selection_mode="multi", 
        default=tags, 
        key='mvp_filter_pills'
    )

    # Create a copy of the data and map 1/0 to the actual text labels for the legend
    plot_data = data.copy()
    plot_data['Player Tier'] = plot_data['mvp_caliber'].apply(lambda x: 'MVP Caliber' if x == 'yes' else 'Rest of League')

    # Dynamic Logic for Filtering and Coloring
    if not selected_tags:
        # Condition A: None selected -> Show whole league as one solid color
        filtered_data = plot_data
        color_col = None
    elif len(selected_tags) == 2:
        # Condition B: Both selected -> Show whole league, but split into two colors
        filtered_data = plot_data
        color_col = 'Player Tier'
    else:
        # Condition C: Exactly one selected -> Filter dataset to just that tier, one solid color
        filtered_data = plot_data[plot_data['Player Tier'] == selected_tags[0]]
        color_col = None

    fig_detailed_hist = px.histogram(
        filtered_data, 
        x=selected_feat, 
        color=color_col,            # Dynamically applies color split if both are selected
        barmode="overlay",          # Overlays the two distributions instead of stacking them
        nbins=40, 
        marginal="box",
        hover_data=['Player', 'Team'], 
        
        color_discrete_map={'MVP Caliber': '#90D743', 'Rest of League': '#477E9F'} if color_col else None,
        color_discrete_sequence=["#31688E"] if not color_col else None
    )
    
    # Reduce opacity slightly so overlaid histograms don't completely hide each other
    fig_detailed_hist.update_traces(opacity=0.75)
    fig_detailed_hist.update_layout(
        bargap=0.05, 
        height=400, # Increased height slightly so the boxplot has room to breathe
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",      # Make legend horizontal
            yanchor="top",
            y=-0.15,              # Push legend below the x-axis
            xanchor="center",
            x=0.5,
            title=None            # Hide the legend title to maximize width
        )
    )

    st.plotly_chart(fig_detailed_hist, use_container_width=True)

st.markdown("---")