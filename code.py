import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="FIFA 23 Stats Visualizer", layout="wide", page_icon="⚽")

st.title("⚽ FIFA 23 Player Stats Visualizer")

# -----------------------------
# Load and Prepare Data
# -----------------------------
@st.cache_data
def load_data():
    # Make sure this filename matches exactly what is in your GitHub repo
    data = pd.read_csv("Fifa 23 Players Data.csv")
    data.dropna(subset=['Known As', 'Overall', 'Pace Total', 'Shooting Total', 'Passing Total', 'Dribbling Total', 'Defending Total', 'Physicality Total'], inplace=True)
    return data

fifa_data = load_data()

features = [
    "Pace Total", "Shooting Total", "Passing Total", 
    "Dribbling Total", "Defending Total", "Physicality Total"
]

# We create a scaled version for the ML model, but keep the original for display/charts
scaler = StandardScaler()
fifa_data_scaled = fifa_data.copy()
fifa_data_scaled[features] = scaler.fit_transform(fifa_data_scaled[features])

# -----------------------------
# Train ML Model
# -----------------------------
@st.cache_resource
def train_model():
    X = fifa_data_scaled[features]
    y = fifa_data_scaled["Overall"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae

model, mae = train_model()
st.sidebar.info(f"Model Training Complete\nMAE: {mae:.2f}")

# -----------------------------
# Sidebar Menu
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "Filter Players",
        "Top 5 Players",
        "Player Comparison (Radar)",
        "Passing vs Shooting",
        "Predict Player Performance"
    ]
)

# -----------------------------
# 1. Filter Players
# -----------------------------
if menu == "Filter Players":
    st.header("🔍 Filter Players by Attribute")
    
    col1, col2 = st.columns(2)
    with col1:
        attribute = st.selectbox("Select Attribute to Filter", ["Overall", "Age", "Potential", "Height(in cm)", "Weight(in kg)"])
    with col2:
        min_val = int(fifa_data[attribute].min())
        max_val = int(fifa_data[attribute].max())
        val_range = st.slider(f"Select {attribute} Range", min_val, max_val, (min_val + 10, max_val))

    filtered_df = fifa_data[(fifa_data[attribute] >= val_range[0]) & (fifa_data[attribute] <= val_range[1])]
    
    st.write(f"Found {len(filtered_df)} players matching your criteria.")
    # Replace line 90 with this safe version:
desired_cols = ["Known As", "Full Name", "Nationality", "Club", attribute]
# This only selects columns that actually exist in the file
existing_cols = [c for c in desired_cols if c in filtered_df.columns]

st.dataframe(filtered_df[existing_cols].sort_values(by=attribute, ascending=False))

# -----------------------------
# 2. Top 5 Players
# -----------------------------
elif menu == "Top 5 Players":
    st.header("📊 Top 5 Players by Attribute")
    
    attribute = st.selectbox("Select Attribute", features + ["Overall", "Potential"])
    top_players = fifa_data.nlargest(5, attribute)[["Known As", attribute]]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_players, x="Known As", y=attribute, ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -----------------------------
# 3. Radar Chart Comparison
# -----------------------------
elif menu == "Player Comparison (Radar)":
    st.header("📈 Compare Two Players")
    
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Select Player 1", fifa_data["Known As"].unique())
    with col2:
        p2 = st.selectbox("Select Player 2", fifa_data["Known As"].unique(), index=1)

    def draw_radar(player1, player2):
        # Using unscaled data for the chart so labels make sense (e.g. 0-100)
        v1 = fifa_data[fifa_data["Known As"] == player1][features].values.flatten()
        v2 = fifa_data[fifa_data["Known As"] == player2][features].values.flatten()
        
        # Close the loop for the radar chart
        v1 = np.append(v1, v1[0])
        v2 = np.append(v2, v2[0])
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, v1, color='blue', label=player1)
        ax.fill(angles, v1, color='blue', alpha=0.2)
        ax.plot(angles, v2, color='red', label=player2)
        ax.fill(angles, v2, color='red', alpha=0.2)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        return fig

    st.pyplot(draw_radar(p1, p2))

# -----------------------------
# 4. Scatter Plot
# -----------------------------
elif menu == "Passing vs Shooting":
    st.header("⚡ Passing vs Shooting Correlation")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=fifa_data, x="Passing Total", y="Shooting Total", hue="Overall", palette="magma", alpha=0.6)
    st.pyplot(fig)

# -----------------------------
# 5. Prediction
# -----------------------------
elif menu == "Predict Player Performance":
    st.header("🤖 Predict Player Overall Rating")
    st.write("This model uses Random Forest to predict a player's Overall rating based on their 6 main stats.")

    player_name = st.selectbox("Choose a Player to Predict", fifa_data["Known As"].unique())
    
    if st.button("Run Prediction"):
        # We must use the SCALED data for the model input
        player_row_scaled = fifa_data_scaled[fifa_data_scaled["Known As"] == player_name][features]
        actual_val = fifa_data[fifa_data["Known As"] == player_name]["Overall"].values[0]
        
        prediction = model.predict(player_row_scaled)[0]
        
        col1, col2 = st.columns(2)
        col1.metric("Actual Overall", f"{actual_val}")
        col2.metric("Predicted Overall", f"{prediction:.2f}")
        
        st.info(f"Difference: {abs(actual_val - prediction):.2f}")
