import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="2026 WNBA MVP Race Predictions", layout="wide")

# Load Data with caching for better performance
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
@st.cache_data
def load_data():
    data = pd.read_csv(PROJECT_ROOT / 'ml' / 'data'/ 'predictions.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    
    data['mvp_caliber'] = np.where(data['Predicted_Share'] >= 0.1 , 'yes', 'no')
    return data

try:
    data = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'data/predictions.csv' exists.")
    st.stop()

st.title("2026 WNBA MVP Race Predictions & Analytics 🏀 ")
    

st.markdown("---")
leader_name = data.iloc[0]['Player']

# --- ROW 1: MVP Frontrunner & TOP 5 MVP LEADERS  ---
col1, col2 = st.columns([1, 2])


with col1:
    
    st.subheader("🏆 MVP Frontrunner")
    leader_share = data.iloc[0]['Predicted_Share']
    st.metric(label="",value=leader_name, delta=f"{leader_share:.2%} Pred. Share")

with col2:

    st.subheader("Top 5 MVP Contenders")
    st.markdown("Current leaders based on predicted award share:")

    top_5 = data[['Player', 'Team', 'Pos', 'Predicted_Share']].head(5)
    
    st.dataframe(
        top_5.style.format({'Predicted_Share': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )

features = [
    ('USG%', 'Usage Percentage (USG%)'),
    ('Predicted_Share', 'Predicted Share'),
    ('WS', 'Win Shares (WS)'),
    ('PER', 'Player Efficiency Rating (PER)'),
    ('TS%', 'True Shooting % (TS%)'),
    ('ast_per_game', 'Assists Per Game'),
    ('trb_per_game', 'Total Rebounds Per Game'),
    ('blk_per_game', 'Blocks Per Game'),
    ('stl_per_game', 'Steals Per Game')
]

# --- ROW 2: HEATMAP AND INTERACTIVE SCATTER PLOTS  ---
st.markdown("---")
col3, col4 = st.columns([1,1])

with col3:

    st.subheader("Which Stats Matter Most to the Model?")
    st.markdown("The statistics most closely linked to a high predicted award share")
    
    cols_to_correlate = data.columns[4:14]
    corr_matrix = data[cols_to_correlate].corr(method='spearman')
    
    # 2. Extract correlation with Predicted_Share, remove self-correlation, and sort
    corr_with_target = corr_matrix['Predicted_Share'].drop('Predicted_Share').sort_values(ascending=False)
    
    # 3. Convert the Series into a DataFrame for Plotly Express
    corr_df = corr_with_target.reset_index()
    corr_df.columns = ['Feature', 'Correlation']
    
    # 4. Create the Interactive Plotly Bar Chart
    fig_corr_bar = px.bar(
        corr_df,
        x='Feature',
        y='Correlation',
        color='Correlation',              # This creates the continuous color gradient!
        color_continuous_scale='viridis', # The colormap you requested
        text_auto='.2f',                  # Puts the exact correlation value on top of each bar
        labels={'Correlation': 'Spearman Correlation'}
    )
    
    # 5. Layout optimizations (mobile-friendly and clean)
    fig_corr_bar.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), 
        height=320,
        xaxis=dict(tickangle=-45, title=None), # Angles feature names so they don't overlap
        coloraxis_showscale=False              # Hides the color legend to save space (since Y-axis already shows the value)
    )
    
    st.plotly_chart(fig_corr_bar, use_container_width=True)

with col4:

    st.subheader(f"Where {leader_name} Stands Among the League")
    st.markdown("Comparing players by points per game and usage rate — hover for player details, or switch the Y-axis to explore other stats")

    numeric_columns = [f[1] for f in features]
    selected_feat_label = st.selectbox("Select a metric to explore:", options=numeric_columns, index=0, key='metric_filter')
    selected_feat = [f[0] for f in features if f[1] == selected_feat_label][0]

    fig = px.scatter(
        data, y=selected_feat, x='pts_per_game',
        color='Predicted_Share', hover_name='Player',
        hover_data=['Team', 'Pos', 'Predicted_Share'],
        color_continuous_scale='viridis',
        labels={'pts_per_game': 'Points Per Game',selected_feat: selected_feat_label}
    )
    fig.update_layout(
            margin=dict(l=10, r=10, t=25, b=10), 
            height=320, # Slightly taller to make room for the bottom colorbar
            coloraxis_colorbar=dict(
                orientation="h",    # Make colorbar horizontal
                yanchor="top",
                y=-0.25,            # Push it below the x-axis
                xanchor="center",
                x=0.5,
                thickness=10,       # Make the bar slimmer
                title=""            # Hide the 'Predicted_Share' title to save space
            )
        )
    st.plotly_chart(fig, use_container_width=True)
    
st.markdown("---")

    # --- ROW 3: FEATURE DISTRIBUTIONS ---

with st.container():
    st.subheader("MVP-Caliber Players vs. the Rest of the League")
    st.markdown(f"How the top 9 MVP-caliber players — and {leader_name} in particular — separate from the rest of the league")

    # 1. Feature selection (Keep at the top)
    features_ = features
    features_.insert(0,('pts_per_game', 'Points per Game'))
    numeric_columns = [f[1] for f in features_]
    selected_feat_label = st.selectbox("Select a metric to explore:", options=numeric_columns, index=0)
    selected_feat = [f[0] for f in features_ if f[1] == selected_feat_label][0]

    # 2. Setup Pills for MULTIPLE selections
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

    # 3. Dynamic Logic for Filtering and Coloring
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

    # 4. Create the Histogram
    fig_detailed_hist = px.histogram(
        filtered_data, 
        x=selected_feat, 
        color=color_col,            # Dynamically applies color split if both are selected
        barmode="overlay",          # Overlays the two distributions instead of stacking them
        nbins=40, 
        marginal="box",
        hover_data=['Player', 'Team'], 
        
        # Custom color mappings: MVP (Gold/Orange), Rest of League (Blue). Solid Blue if single tag.
        color_discrete_map={'MVP Caliber': '#90D743', 'Rest of League': '#477E9F'} if color_col else None,
        color_discrete_sequence=["#31688E"] if not color_col else None
    )
    
    # Reduce opacity slightly so overlaid histograms don't completely hide each other
    fig_detailed_hist.update_traces(opacity=0.75)
    fig_detailed_hist.update_layout(
        bargap=0.05, 
        height=450, # Increased height slightly so the boxplot has room to breathe
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