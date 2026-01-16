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
