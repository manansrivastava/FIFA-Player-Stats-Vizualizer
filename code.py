import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load FIFA 23 Dataset
file_path = "Fifa 23 Players Data.csv"  # Update with correct path if needed
fifa_data = pd.read_csv(file_path)

# Data Preprocessing
fifa_data.dropna(inplace=True)  # Remove missing values
scaler = StandardScaler()

# Select Relevant Features
features = ["Pace Total", "Shooting Total", "Passing Total", "Dribbling Total", "Defending Total", "Physicality Total"]
fifa_data[features] = scaler.fit_transform(fifa_data[features])

# 🔹 Function to Display Attribute List for User Selection
def select_attribute():
    available_attributes = fifa_data.columns.tolist()
    print("\nAvailable Attributes:")
    for i, attr in enumerate(available_attributes):
        print(f"{i+1}. {attr}")
    
    while True:
        try:
            choice = int(input("\nEnter the number corresponding to your chosen attribute: "))
            if 1 <= choice <= len(available_attributes):
                return available_attributes[choice - 1]
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# 🔹 Function to Get Players Based on Attribute (e.g., Nationality = France)
def get_players_by_attribute():
    attribute = select_attribute()
    value = input(f"Enter value for {attribute} (e.g., France): ")
    filtered_players = fifa_data[fifa_data[attribute] == value]["Known As"].tolist()
    
    if not filtered_players:
        print(f"No players found with {attribute} = {value}")
    else:
        print(f"Players with {attribute} = {value}:")
        for player in filtered_players:
            print(player)
    return filtered_players

# 🔹 Function to Compare Top 5 Players Based on an Attribute (Bar Chart)
def plot_top_players(attribute):
    top_players = fifa_data.nlargest(5, attribute)[["Known As", attribute]]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_players["Known As"], y=top_players[attribute], palette="coolwarm")
    plt.title(f"Top 5 Players by {attribute}")
    plt.xlabel("Players")
    plt.ylabel(attribute)
    plt.xticks(rotation=30)
    plt.show()

# 🔹 Function to Compare Two Players Using Radar Chart
def plot_radar_chart_comparison(player1_name, player2_name):
    player1 = fifa_data[fifa_data["Known As"] == player1_name]
    player2 = fifa_data[fifa_data["Known As"] == player2_name]
    
    if player1.empty or player2.empty:
        print("One or both players not found. Please select valid players.")
        return
    
    player1 = player1.iloc[0]
    player2 = player2.iloc[0]
    
    values1 = [player1[feat] for feat in features]
    values2 = [player2[feat] for feat in features]
    values1 += values1[:1]
    values2 += values2[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values1, color="blue", alpha=0.4, label=player1_name)
    ax.fill(angles, values2, color="red", alpha=0.4, label=player2_name)
    ax.plot(angles, values1, color="blue", linewidth=2)
    ax.plot(angles, values2, color="red", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    plt.title(f"Comparison: {player1_name} vs {player2_name}")
    plt.legend()
    plt.show()

# 🔹 Function to Plot Scatter Plot for Passing vs. Shooting
def plot_scatter():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=fifa_data, x="Passing Total", y="Shooting Total", hue="Overall", size="Overall", palette="coolwarm")
    plt.title("Passing vs. Shooting Performance (FIFA 23 Players)")
    plt.xlabel("Passing Total")
    plt.ylabel("Shooting Total")
    plt.show()

# 🔹 Train Machine Learning Model for Player Performance Prediction
def train_performance_model():
    target = "Overall"
    X = fifa_data[features]
    y = fifa_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Regression Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    return model

# 🔹 Predict a Player's Future Performance
def predict_player_performance(model):
    player_name = input("Enter player name: ")
    player = fifa_data[fifa_data["Known As"] == player_name]
    
    if player.empty:
        print("Player not found. Please enter a valid name.")
        return
    
    features_values = player[features].values
    features_df = pd.DataFrame(features_values, columns=features)  # Fix sklearn warning
    predicted_overall = model.predict(features_df)[0]
    print(f"Predicted Future Overall Rating for {player_name}: {predicted_overall:.2f}")

# 🔹 Interactive Menu
def main_menu():
    model = train_performance_model()
    while True:
        print("\nMENU:")
        print("1. Filter Players by Attribute")
        print("2. Compare Top 5 Players in an Attribute")
        print("3. Compare Two Players (Radar Chart)")
        print("4. Show Scatter Plot (Passing vs. Shooting)")
        print("5. Predict Player Future Performance")
        print("6. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            get_players_by_attribute()
        elif choice == "2":
            attribute = select_attribute()
            plot_top_players(attribute)
        elif choice == "3":
            player1_name = input("Enter first player name: ")
            player2_name = input("Enter second player name: ")
            plot_radar_chart_comparison(player1_name, player2_name)
        elif choice == "4":
            plot_scatter()
        elif choice == "5":
            predict_player_performance(model)
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

# Run the interactive menu
if __name__ == "__main__":
    main_menu()
