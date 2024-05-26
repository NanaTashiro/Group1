import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Function to train and evaluate models
def train_and_evaluate_models(X_train_scaled_df, Y_train):
    # Decision Tree Regressor
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train_scaled_df, Y_train)
    tree_pred = tree_model.predict(X_train_scaled_df)
    tree_mse = mean_squared_error(Y_train, tree_pred)
    tree_r2 = r2_score(Y_train, tree_pred)
    tree_mae = mean_absolute_error(Y_train, tree_pred)

    st.write("Decision Tree Regressor:")
    st.write("Mean Squared Error:", tree_mse)
    st.write("R-squared:", tree_r2)
    st.write("Mean Absolute Error:", tree_mae)
    st.write()

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled_df, Y_train)
    linear_pred = linear_model.predict(X_train_scaled_df)
    linear_mse = mean_squared_error(Y_train, linear_pred)
    linear_r2 = r2_score(Y_train, linear_pred)
    linear_mae = mean_absolute_error(Y_train, linear_pred)

    st.write("Linear Regression:")
    st.write("Mean Squared Error:", linear_mse)
    st.write("R-squared:", linear_r2)
    st.write("Mean Absolute Error:", linear_mae)
    st.write()

    # Polynomial Regression
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train_scaled_df)
    X_valid_poly = poly_features.transform(X_train_scaled_df)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Y_train)
    poly_pred = poly_model.predict(X_valid_poly)
    poly_mse = mean_squared_error(Y_train, poly_pred)
    poly_r2 = r2_score(Y_train, poly_pred)
    poly_mae = mean_absolute_error(Y_train, poly_pred)

    st.write("Polynomial Regression:")
    st.write("Mean Squared Error:", poly_mse)
    st.write("R-squared:", poly_r2)
    st.write("Mean Absolute Error:", poly_mae)
    st.write()

    # Random Forest Regressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_scaled_df, Y_train)

    y_pred = model.predict(X_train_scaled_df)
    mae = mean_absolute_error(Y_train, y_pred)
    rf_pred = model.predict(X_train_scaled_df)
    rf_mse = mean_squared_error(Y_train, rf_pred)
    rf_r2 = r2_score(Y_train, rf_pred)

    st.write("Random Forest Regressor:")
    st.write("Mean Squared Error:", rf_mse)
    st.write("R-squared:", rf_r2)
    st.write("Mean Absolute Error:", mae)
    st.write()

    return tree_pred

# Function to plot actual vs predicted votes
def plot_actual_vs_predicted_by_electorate_year(electorate, year, actual_df, predicted_df):
    actual_data = actual_df[(actual_df['Electorate'] == electorate) & (actual_df['Election Year'] == year)]
    predicted_data = predicted_df[(predicted_df['Electorate'] == electorate) & (predicted_df['Election Year'] == year)]

    if actual_data.empty or predicted_data.empty:
        st.write(f"No data available for {electorate} in {year}")
        return

    actual_values = actual_data.values[0][2:]  # Exclude 'Electorate' and 'Election Year'
    predicted_values = predicted_data.values[0][2:]  # Exclude 'Electorate' and 'Election Year'
    parties = actual_data.columns[2:]  # Exclude 'Electorate' and 'Election Year'

    plt.figure(figsize=(10, 6))
    x = np.arange(len(parties))
    plt.scatter(x, actual_values, color='blue', label='Actual', marker='o')
    plt.scatter(x, predicted_values, color='red', label='Predicted', marker='x')
    plt.title(f"Actual vs. Predicted Votes for {electorate} - {year}")
    plt.xlabel("Party")
    plt.ylabel("Votes")
    plt.xticks(x, parties, rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot polling results over time
def plot_polling_results(df, parties):
    plt.figure(figsize=(15, 8))
    for party in parties:
        plt.plot(df.index, df[party], label=party)
    plt.title('Polling Results Over Time')
    plt.xlabel('Date')
    plt.ylabel('Polling Percentage')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside the plot
    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    st.pyplot(plt)

# Load data
@st.cache_data
def load_data():
    election_poll_2017_2024 = pd.read_csv('election_poll_2017_2024.csv', index_col='Date', parse_dates=True)
    return election_poll_2017_2024

# Main Streamlit app
def main():
    st.title('New Zealand Election Polling Data Analysis')
    st.write('This app visualizes the polling results of various political parties in New Zealand from 2017 to 2024.')

    # Load data
    df = load_data()

    # Convert columns to numeric
    parties = ['National Party', 'Labour Party', 'Green Party', 'New Zealand First Party', 'ACT New Zealand', 'Other']
    for party in parties:
        df[party] = pd.to_numeric(df[party], errors='coerce')

    # Fill NaN values
    df.fillna(method='ffill', inplace=True)

    # Select parties to display
    selected_parties = st.multiselect('Select parties to display', options=parties, default=parties)

    if selected_parties:
        plot_polling_results(df, selected_parties)
    else:
        st.write('Please select at least one party to display.')

    # Model training and evaluation
    st.write('### Model Training and Evaluation')
    X_train_scaled_df = pd.read_csv('X_train_scaled_df.csv')  # Replace with actual data loading
    Y_train = pd.read_csv('Y_train.csv')  # Replace with actual data loading
    tree_pred = train_and_evaluate_models(X_train_scaled_df, Y_train)
    
    # Creating a DataFrame for predictions
    tree_pred_df = pd.DataFrame(tree_pred, columns=Y_train.columns, index=Y_train.index)
    
    # Plot actual vs predicted votes
    unique_electorates = Y_train['Electorate'].unique()
    for electorate in unique_electorates:
        plot_actual_vs_predicted_by_electorate_year(electorate, 2017, Y_train, tree_pred_df)

# Run the app
if __name__ == '__main__':
    main()
