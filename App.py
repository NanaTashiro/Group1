import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import leafmap.foliumap as leafmap
import plotly.graph_objects as go
from PIL import Image

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Introduction", "Data Collection", "Regression Models",  "KNN Regression", "Neural Network"])

def show_intro_page():
    st.title("Predicting Election Results (Party Lists) for the Auckland Region")
    
    st.header("Group 1")
    st.write("""
    - Nana (Nuthita) Tashiro, ID: 21016134
    - Cole Palffy, ID: 18047471
    - Nazgul Altynbekova, ID: 22012935
    """)
    
    st.header("Research Question")
    st.write("""
    How can an integrated prediction model combining different techniques accurately predict the next election results of four major parties and others (ACT New Zealand, Green Party, Labour Party, National Party, Other) for every electorate within the Auckland region if the parliament were dissolved today?
    """)

def show_Data_Collection_page():
    st.title("Data Collection")

    st.header("Election Results (List) of each electorate in Auckland region from 2017-2023")
    try:
        combined_result_list = pd.read_csv('combined_result_list.csv')
        st.dataframe(combined_result_list)

        party_colors = {
            'ACT New Zealand Vote': 'yellow',
            'Green Party Vote': 'green',
            'Labour Party Vote': 'red',
            'National Party Vote': 'blue',
            'New Zealand First Party Vote': 'black',
            'Others Vote': 'grey'
        }

        st.write("Select parties to visualize:")
        selected_parties = st.multiselect('Parties', options=list(party_colors.keys()), default=list(party_colors.keys()))

        if selected_parties:
            party_votes_corrected = combined_result_list.groupby('Election Year')[selected_parties].mean(numeric_only=True)
            
            fig = go.Figure()
            for column in party_votes_corrected.columns:
                fig.add_trace(go.Scatter(
                    x=party_votes_corrected.index, 
                    y=party_votes_corrected[column], 
                    mode='lines+markers', 
                    name=column, 
                    line=dict(color=party_colors.get(column))
                ))
            
            fig.update_layout(
                title='Average Party Vote Percentages in Auckland Region (2017-2023)',
                xaxis_title='Election Year',
                yaxis_title='Average Party Vote by Percentage',
                legend_title='Party',
                hovermode='x unified'
            )
            st.plotly_chart(fig)
        else:
            st.write("Please select at least one party to visualize.")
    except FileNotFoundError:
        st.error("The file 'combined_result_list.csv' was not found. Please upload the file to proceed.")

    st.header("Demographic Data of all electorates from 2017-2024")
    try:
        combined_all_demo = pd.read_csv('combined_all_demo.csv')
        st.dataframe(combined_all_demo)
    except FileNotFoundError:
        st.error("The file 'combined_all_demo.csv' was not found. Please upload the file to proceed.")
    
    try:
        Area_Coords = pd.read_csv('Geo_info.csv')
        Area_Coords = Area_Coords[['Respondents','Latitude', 'Longitude', '15-29 years',
               '30-64 years', '65 years and over', 'No qualification',
               'Level 3 certificate', 'Level 4 certificate', 'Level 5 diploma', 'Level 6 diploma',
               'Bachelor degree and level 7 qualification',
               'Post-graduate and honours degrees', 'Masters degree',
               'Doctorate degree',
               'European', 'Maori',
               'Pacific Peoples', 'Asian', 'Middle Eastern/Latin American/African',
               'Other ethnicity', 'Regular smoker', 'Ex-smoker',
               'Never smoked regularly', 'No dependent children',
               'One dependent child', 'Two dependent children',
               'Three dependent children', 'Four or more dependent children',
               'Total people - with at least one religious affiliation', 'No religion']]

        st.title("Marker Cluster")
        m = leafmap.Map(center=[-36.8, 175], zoom=8.5)
        m.add_points_from_xy(
            Area_Coords,
            x="Longitude",
            y="Latitude",
            icon_names=['gear', 'map', 'leaf', 'globe'],
            spin=True,
            add_legend=True,
        )
        m.to_streamlit(height=650)
    except FileNotFoundError:
        st.error("The file 'Geo_info.csv' was not found. Please upload the file to proceed.")

    st.header("Polling data from 2017-2024")
    try:
        election_poll_2017_2024 = pd.read_csv('election_poll_2017_2024.csv')
        st.dataframe(election_poll_2017_2024)

        # Convert the notebook code into Streamlit interactive model
        election_poll_2017_2024['Date'] = pd.to_datetime(election_poll_2017_2024['Date'])
        election_poll_2017_2024.sort_values(by='Date', inplace=True)

        parties = ['National Party', 'Labour Party', 'Green Party', 'New Zealand First Party', 'ACT New Zealand', 'Other']
        for party in parties:
            election_poll_2017_2024[f'{party} Change'] = election_poll_2017_2024[party].diff()

        key_events = pd.read_csv('key_events.csv')
        key_events['Date'] = pd.to_datetime(key_events['Date'])
        key_events.sort_values(by='Date', inplace=True)

        significant_changes = election_poll_2017_2024.melt(id_vars=['Date', 'Poll'], 
                                                           value_vars=[f'{party} Change' for party in parties], 
                                                           var_name='Party', 
                                                           value_name='Change')
        significant_changes = significant_changes[significant_changes['Change'].abs() > 5]
        impactful_events = []

        for _, row in significant_changes.iterrows():
            event_window_start = row['Date'] - pd.Timedelta(days=7)
            event_window_end = row['Date']
            
            relevant_events = key_events[(key_events['Date'] >= event_window_start) & (key_events['Date'] <= event_window_end)]
            
            for _, event_info in relevant_events.iterrows():
                impactful_events.append({
                    'Date': row['Date'],
                    'Party': row['Party'].replace(' Change', ''),
                    'Change': row['Change'],
                    'Event Date': event_info['Date'],
                    'Event': event_info['Event']
                })

        impactful_events_df = pd.DataFrame(impactful_events)
        impactful_events_df['Event Date'] = pd.to_datetime(impactful_events_df['Event Date'])
        combined_events = impactful_events_df.groupby(['Event Date', 'Event']).apply(lambda df: ', '.join([f"{row['Party']} ({row['Change']:+.1f}%)" for _, row in df.iterrows()])).reset_index()
        combined_events.columns = ['Event Date', 'Event', 'Impact']
        
        fig = go.Figure()
        for party in parties:
            fig.add_trace(go.Scatter(
                x=election_poll_2017_2024['Date'], 
                y=election_poll_2017_2024[party], 
                mode='lines+markers', 
                name=party
            ))
        
        for _, row in impactful_events_df.iterrows():
            fig.add_vline(x=row['Date'], line=dict(color='gray', width=1, dash='dash'))
        
        fig.update_layout(
            title='Polling Results with Changes > 5%',
            xaxis_title='Date',
            yaxis_title='Polling Percentage',
            legend_title='Party',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
    except FileNotFoundError:
        st.error("The file 'election_poll_2017_2024.csv' was not found. Please upload the file to proceed.")

def show_regression_page():
    st.title("Linear, Polynomial, Decision Tree and Random Forest Regression")
    st.subheader("First Try")

    st.write("""
    Random Forest Regressor:
    - Mean Squared Error: 61.00338034404762
    - R-squared: 0.43813108642746235
    - Mean Absolute Error: 4.686294047619047

    Decision Tree Regressor:
    - Mean Squared Error: 102.93514940476187
    - R-squared: -0.05440366853861641
    - Mean Absolute Error: 5.632678571428571

    Linear Regression:
    - Mean Squared Error: 2.9921841476839266e+26
    - R-squared: -1.5004313073007013e+25
    - Mean Absolute Error: 7743677865302.306

    Polynomial Regression:
    - Mean Squared Error: 55.23818229929456
    - R-squared: 0.36154597596526533
    - Mean Absolute Error: 4.163687333057046
    """)

    image1 = Image.open('Regression Model Performance Metrics.png')
    st.image(image1, caption='Regression Model Performance Metrics', use_column_width=True)

    st.subheader("Feature Selection")

    st.write("""
    Random Forest Regressor:
    - Mean Squared Error: 61.529651116190486
    - R-squared: 0.47229381455511765
    - Mean Absolute Error: 4.648752380952382

    Decision Tree Regressor:
    - Mean Squared Error: 150.70449285714287
    - R-squared: -0.3545009386089091
    - Mean Absolute Error: 6.989404761904763

    Linear Regression:
    - Mean Squared Error: 300.8785321014947
    - R-squared: -1.122585072938829
    - Mean Absolute Error: 8.548956684528763

    Polynomial Regression:
    - Mean Squared Error: 64.62478041132772
    - R-squared: 0.21470107103855648
    - Mean Absolute Error: 4.9548058986689805
    """)

    image2 = Image.open('Metrics after Feature Selection.png')
    st.image(image2, caption='Metrics after Feature Selection', use_column_width=True)

    st.title("Actual vs Predicted Model")

    num_images = 21  
    for i in range(num_images):
        image_path = f'plot{i}.png'
        image = Image.open(image_path)
        st.image(image, caption=f'Plot {i}', use_column_width=True)
        
        
def show_knn_page():
    # Subset of features (party votes)
    subset_features = ['National Party Vote', 'Labour Party Vote', 'Green Party Vote', 'New Zealand First Party Vote', 'ACT New Zealand Vote', 'Others Vote']
    
    # Defining party colors
    party_colors = {
        'ACT New Zealand Vote': 'yellow',
        'Green Party Vote': 'green',
        'Labour Party Vote': 'red',
        'National Party Vote': 'blue',
        'New Zealand First Party Vote': 'black',
        'Others Vote': 'grey'
    }
    
    # Set the title of the Streamlit app
    st.title("KNN Model")
    
    # Load datasets
    new_merged_demo_polls_path = 'merged_demo_polls.csv'
    new_combined_result_list_path = 'combined_result_list.csv'
    
    new_merged_demo_polls = pd.read_csv(new_merged_demo_polls_path)
    new_combined_result_list = pd.read_csv(new_combined_result_list_path)
    
    # Prepare the training set (2017, 2020, 2023) and the prediction set (2024)
    combined_data_train = pd.concat([new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2017],
                                     new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2020],
                                     new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2023]])
    
    combined_targets_train = pd.concat([new_combined_result_list[new_combined_result_list['Election Year'] == 2017],
                                        new_combined_result_list[new_combined_result_list['Election Year'] == 2020],
                                        new_combined_result_list[new_combined_result_list['Election Year'] == 2023]])
    
    # Prepare the feature set for 2024 prediction
    prediction_data = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2024]
    
    # Splitting the data into features (X) and targets (Y)
    X_train = combined_data_train.drop(columns=['Election Year', 'Electorate'])
    Y_train = combined_targets_train.drop(columns=['Election Year', 'Electorate'])
    X_test = prediction_data.drop(columns=['Election Year', 'Electorate'])
    
    # Normalize and standardize data
    min_max_scaler = MinMaxScaler()
    X_train_normalized = min_max_scaler.fit_transform(X_train)
    X_test_normalized = min_max_scaler.transform(X_test)
    
    standard_scaler = StandardScaler()
    X_train_standardized = standard_scaler.fit_transform(X_train)
    X_test_standardized = standard_scaler.transform(X_test)
    
    # Function to train and evaluate KNN model
    def evaluate_knn(X_train, Y_train, X_test, k):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, Y_train)
        predictions = knn.predict(X_test)
        return predictions
    
    # Function to perform cross-validation and return the mean RMSE
    def cross_val_rmse(model, X, y, cv=5):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
        rmse_scores = np.sqrt(-scores)
        return rmse_scores.mean()
    
    # Evaluate KNN with normalized and standardized data
    k_values = range(1, 50)
    
    normalized_rmse_scores = []
    for k in k_values:
        predictions = evaluate_knn(X_train_normalized, Y_train, X_test_normalized, k)
        rmse = sqrt(mean_squared_error(Y_train, evaluate_knn(X_train_normalized, Y_train, X_train_normalized, k)))
        normalized_rmse_scores.append(rmse)
    
    standardized_rmse_scores = []
    for k in k_values:
        predictions = evaluate_knn(X_train_standardized, Y_train, X_test_standardized, k)
        rmse = sqrt(mean_squared_error(Y_train, evaluate_knn(X_train_standardized, Y_train, X_train_standardized, k)))
        standardized_rmse_scores.append(rmse)
    
    # Plot RMSE values for both normalization and standardization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(k_values, normalized_rmse_scores, marker='o', label='Normalized RMSE')
    ax.plot(k_values, standardized_rmse_scores, marker='o', label='Standardized RMSE')
    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('RMSE')
    ax.set_title('KNN Regression RMSE Comparison')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Determine the best k for both methods
    best_k_normalized = k_values[np.argmin(normalized_rmse_scores)]
    best_k_standardized = k_values[np.argmin(standardized_rmse_scores)]
    st.write(f"Best k for normalized data: {best_k_normalized} with RMSE: {min(normalized_rmse_scores)}")
    st.write(f"Best k for standardized data: {best_k_standardized} with RMSE: {min(standardized_rmse_scores)}")
    
    # Cross-validate RMSE values
    normalized_rmse_cv_scores = []
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        rmse_cv = cross_val_rmse(knn, X_train_normalized, Y_train)
        normalized_rmse_cv_scores.append(rmse_cv)
    
    standardized_rmse_cv_scores = []
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        rmse_cv = cross_val_rmse(knn, X_train_standardized, Y_train)
        standardized_rmse_cv_scores.append(rmse_cv)
    
    # Plot cross-validated RMSE values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(k_values, normalized_rmse_cv_scores, marker='o', label='Normalized RMSE (CV)')
    ax.plot(k_values, standardized_rmse_cv_scores, marker='o', label='Standardized RMSE (CV)')
    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('Cross-Validated RMSE')
    ax.set_title('KNN Regression Cross-Validated RMSE Comparison')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Determine the best k for both methods based on cross-validation
    best_k_normalized_cv = k_values[np.argmin(normalized_rmse_cv_scores)]
    best_k_standardized_cv = k_values[np.argmin(standardized_rmse_cv_scores)]
    st.write(f"Best k for normalized data (CV): {best_k_normalized_cv} with RMSE: {min(normalized_rmse_cv_scores)}")
    st.write(f"Best k for standardized data (CV): {best_k_standardized_cv} with RMSE: {min(standardized_rmse_cv_scores)}")
    
    # Compare uniform and distance weighting for best k
    k = best_k_normalized_cv
    knn_uniform = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    rmse_cv_uniform = cross_val_rmse(knn_uniform, X_train_normalized, Y_train)
    
    knn_distance = KNeighborsRegressor(n_neighbors=k, weights='distance')
    rmse_cv_distance = cross_val_rmse(knn_distance, X_train_normalized, Y_train)
    
    st.write(f"Cross-Validated RMSE for KNN (uniform, k={k}): {rmse_cv_uniform}")
    st.write(f"Cross-Validated RMSE for KNN (distance, k={k}): {rmse_cv_distance}")
    
    if rmse_cv_uniform < rmse_cv_distance:
        final_knn_model = knn_uniform
        best_weights = 'uniform'
    else:
        final_knn_model = knn_distance
    
    # Train the final model on the entire training data
    final_knn_model.fit(X_train_normalized, Y_train)
    
    # Predictions for 2024
    final_predictions_2024 = final_knn_model.predict(X_test_normalized)
    final_predictions_2024_df = pd.DataFrame(final_predictions_2024, columns=Y_train.columns)
    final_predictions_2024_df['Election Year'] = 2024
    final_predictions_2024_df['Electorate'] = prediction_data['Electorate'].values
    
    # Save the 2024 predictions to CSV
    final_predictions_2024_df.to_csv('final_knn_predictions_2024.csv', index=False)
    
    # Prepare data for past predictions
    X_2017 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2017].drop(columns=['Election Year', 'Electorate'])
    X_2020 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2020].drop(columns=['Election Year', 'Electorate'])
    X_2023 = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2023].drop(columns=['Election Year', 'Electorate'])
    
    # Normalize the data
    X_2017_normalized = min_max_scaler.transform(X_2017)
    X_2020_normalized = min_max_scaler.transform(X_2020)
    X_2023_normalized = min_max_scaler.transform(X_2023)
    
    # Make predictions for each year
    predictions_2017 = final_knn_model.predict(X_2017_normalized)
    predictions_2020 = final_knn_model.predict(X_2020_normalized)
    predictions_2023 = final_knn_model.predict(X_2023_normalized)
    
    # Combine predictions with election year and electorates
    predictions_2017_df = pd.DataFrame(predictions_2017, columns=Y_train.columns)
    predictions_2017_df['Election Year'] = 2017
    predictions_2017_df['Electorate'] = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2017]['Electorate'].values
    
    predictions_2020_df = pd.DataFrame(predictions_2020, columns=Y_train.columns)
    predictions_2020_df['Election Year'] = 2020
    predictions_2020_df['Electorate'] = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2020]['Electorate'].values
    
    predictions_2023_df = pd.DataFrame(predictions_2023, columns=Y_train.columns)
    predictions_2023_df['Election Year'] = 2023
    predictions_2023_df['Electorate'] = new_merged_demo_polls[new_merged_demo_polls['Election Year'] == 2023]['Electorate'].values
    
    # Combine predictions for all years into a single DataFrame
    all_predictions_df = pd.concat([predictions_2017_df, predictions_2020_df, predictions_2023_df])
    
    # Reorder columns to place 'Election Year' and 'Electorate' at the front
    all_predictions_df = all_predictions_df[['Election Year', 'Electorate'] + list(Y_train.columns)]
    
    # Save combined predictions to CSV
    all_predictions_df.to_csv('combined_knn_predictions.csv', index=False)
    
    # Define the electorate mapping based on inspection
    electorate_mapping = {
        # Add mappings here based on discrepancies found in inspection
    }
    
    # Normalize and map electorate names
    def normalize_and_map_electorate_names(df, mapping):
        df['Electorate'] = df['Electorate'].str.lower().str.strip().replace(mapping)
        return df
    
    new_combined_result_list = normalize_and_map_electorate_names(new_combined_result_list, electorate_mapping)
    all_predictions_df = normalize_and_map_electorate_names(all_predictions_df, electorate_mapping)
    
    # Function to create comparison DataFrame for a single year
    def create_comparison_df(year, electorate):
        actual_df = new_combined_result_list[(new_combined_result_list['Election Year'] == year) & (new_combined_result_list['Electorate'] == electorate)]
        predicted_df = all_predictions_df[(all_predictions_df['Election Year'] == year) & (all_predictions_df['Electorate'] == electorate)]
        
        comparison_df = pd.DataFrame()
        for feature in subset_features:
            actual_values = actual_df[feature].values
            predicted_values = predicted_df[feature].values
            
            temp_df = pd.DataFrame({
                'Feature': feature,
                'Actual': actual_values,
                'Predicted': predicted_values
            })
            comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True)
        return comparison_df
    
    # Function to plot comparison for a single year using Plotly
    def plot_comparison(comparison_df, year, electorate):
        if comparison_df.empty:
            st.write(f"No common electorates for {year}. Skipping plot.")
            return
    
        comparison_df = comparison_df.melt(id_vars=['Feature'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Votes')
    
        fig = go.Figure()
        for type_ in comparison_df['Type'].unique():
            df_filtered = comparison_df[comparison_df['Type'] == type_]
            fig.add_trace(go.Scatter(x=df_filtered['Feature'], y=df_filtered['Votes'], mode='lines+markers', name=type_))
    
        fig.update_layout(
            title=f'Comparison of Actual and Predicted Votes by Party for {electorate} in {year}',
            xaxis_title='Party',
            yaxis_title='Votes',
            legend_title='Type',
            hovermode='x unified'
        )
        st.plotly_chart(fig)
    
    # User inputs for comparison
    st.header("Compare Actual and Predicted Votes")
    year = st.selectbox("Select Year", [2017, 2020, 2023])
    electorate = st.selectbox("Select Electorate", new_combined_result_list['Electorate'].unique())
    
    comparison_df = create_comparison_df(year, electorate)
    plot_comparison(comparison_df, year, electorate)
    
    # Function to plot 2024 predictions with color by party name as bar charts and add results number using Plotly
    def plot_predictions_2024(predictions_df, electorate):
        predictions = predictions_df[predictions_df['Electorate'] == electorate]
        if predictions.empty:
            st.write(f"No data available for {electorate} in 2024.")
            return
    
        predictions = predictions.melt(id_vars=['Electorate'], value_vars=subset_features, var_name='Party', value_name='Votes')
    
        fig = go.Figure()
        for party in predictions['Party'].unique():
            df_filtered = predictions[predictions['Party'] == party]
            fig.add_trace(go.Bar(x=[party], y=df_filtered['Votes'], name=party, marker_color=party_colors.get(party)))
    
        fig.update_layout(
            title=f'Predicted Votes for {electorate} in 2024',
            xaxis_title='Party',
            yaxis_title='Votes (%)',
            hovermode='x unified'
        )
        
        # Add vote counts on top of each bar
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    
        st.plotly_chart(fig)
    
    # User input for 2024 predictions
    st.header("Predictions for 2024")
    electorate_2024 = st.selectbox("Select Electorate for 2024", prediction_data['Electorate'].unique())
    plot_predictions_2024(final_predictions_2024_df, electorate_2024)



def show_nn_page():
    # Load the data
    final_neural_predictions_2024 = pd.read_csv('final_neural_predictions_2024.csv')
    combined_neural_predictions = pd.read_csv('combined_neural_predictions.csv')
    final_neural_predictions1_2024 = pd.read_csv('final_neural_predictions1_2024.csv')
    combined_result_list = pd.read_csv('combined_result_list.csv')
    
    # Parameters and RMSE
    st.title("Neural Network Model")
    
    st.subheader("Best Parameters from Neural Network Tuning")
    st.write("""
    - Activation: 'relu'
    - Alpha: 0.0001
    - Batch Size: 64
    - Early Stopping: True
    - Hidden Layer Sizes: (50,)
    - Learning Rate: 'constant'
    - Learning Rate Init: 0.1
    - Max Iter: 200
    - Solver: 'adam'
    - normalized data RMSE: 9.284181455443612
    """)
    
    st.subheader("Prediction for 2024 Election")
    st.dataframe(final_neural_predictions_2024)
    
    st.write("""
    Based on the distribution of Labour Party vote predictions for 2024, it appears that the majority of the predictions are clustered at negative. This result is concerning, especially considering the Labour Party's historical significance and strong presence in New Zealand politics.
    """)
    
    def calculate_statistics(df, year, column):
        stats = {
            'Year': year,
            'Count': df[column].count(),
            'Mean': df[column].mean(),
            'Std Dev': df[column].std(),
            'Min': df[column].min(),
            '25th Percentile': df[column].quantile(0.25),
            'Median': df[column].median(),
            '75th Percentile': df[column].quantile(0.75),
            'Max': df[column].max()
        }
        return stats

    # Prepare the actual data for each year
    actual_2017 = combined_result_list[combined_result_list['Election Year'] == 2017]
    actual_2020 = combined_result_list[combined_result_list['Election Year'] == 2020]
    actual_2023 = combined_result_list[combined_result_list['Election Year'] == 2023]
    predictions_2024 = final_neural_predictions_2024

    # Calculate statistics for each year
    stats_2017 = calculate_statistics(actual_2017, 2017, 'Labour Party Vote')
    stats_2020 = calculate_statistics(actual_2020, 2020, 'Labour Party Vote')
    stats_2023 = calculate_statistics(actual_2023, 2023, 'Labour Party Vote')
    stats_2024 = calculate_statistics(predictions_2024, 2024, 'Labour Party Vote')

    # Combine all statistics into a DataFrame
    stats_df = pd.DataFrame([stats_2017, stats_2020, stats_2023, stats_2024])

    # Display the statistics
    st.subheader("Display the statistics the historical data of Labour Party")
    st.dataframe(stats_df)
    
    st.subheader("Cross-validate the Model on a Subset of Historical Data")
    st.write("""
    Cross-validation RMSE scores: [10.58263428, 7.5126416, 9.74489136, 8.80247001, 8.84035454]
    """)
    st.write("""
    Mean cross-validation RMSE: 9.09659835907756
    """)
    
    st.subheader("Combined Neural Prediction for 2017, 2020, 2023")
    st.dataframe(combined_neural_predictions)
    
    st.subheader("Final Neural Prediction for 2024 after Cross-validation")
    st.dataframe(final_neural_predictions1_2024)
    
    # Normalize and map electorate names
    def normalize_and_map_electorate_names(df, mapping):
        df['Electorate'] = df['Electorate'].str.lower().str.strip().replace(mapping)
        return df
    
    electorate_mapping = {
        # Add mappings here if necessary
    }
    
    combined_result_list = normalize_and_map_electorate_names(combined_result_list, electorate_mapping)
    combined_neural_predictions = normalize_and_map_electorate_names(combined_neural_predictions, electorate_mapping)
    
    # Function to create comparison DataFrame for a single year
    def create_comparison_df(year, electorate):
        actual_df = combined_result_list[(combined_result_list['Election Year'] == year) & (combined_result_list['Electorate'] == electorate)]
        predicted_df = combined_neural_predictions[(combined_neural_predictions['Election Year'] == year) & (combined_neural_predictions['Electorate'] == electorate)]
        
        comparison_df = pd.DataFrame()
        for feature in ['ACT New Zealand Vote', 'Green Party Vote', 'Labour Party Vote', 'National Party Vote', 'New Zealand First Party Vote', 'Others Vote']:
            actual_values = actual_df[feature].values
            predicted_values = predicted_df[feature].values
            
            # Handle mismatched lengths by aligning data
            min_length = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_length]
            predicted_values = predicted_values[:min_length]
            
            temp_df = pd.DataFrame({
                'Feature': feature,
                'Actual': actual_values,
                'Predicted': predicted_values
            })
            comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True)
        return comparison_df
    # Define subset_features at the top or within the function
    subset_features = ['ACT New Zealand Vote', 'Green Party Vote', 'Labour Party Vote', 'National Party Vote', 'New Zealand First Party Vote', 'Others Vote']
    # Function to plot comparison for a single year using Plotly
    def plot_comparison(comparison_df, year, electorate):
        if comparison_df.empty:
            st.write(f"No common electorates for {year}. Skipping plot.")
            return
    
        comparison_df = comparison_df.melt(id_vars=['Feature'], value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Votes')
    
        fig = go.Figure()
        for type_ in comparison_df['Type'].unique():
            df_filtered = comparison_df[comparison_df['Type'] == type_]
            fig.add_trace(go.Scatter(x=df_filtered['Feature'], y=df_filtered['Votes'], mode='lines+markers', name=type_))
    
        fig.update_layout(
            title=f'Comparison of Actual and Predicted Votes by Party for {electorate} in {year}',
            xaxis_title='Party',
            yaxis_title='Votes',
            xaxis=dict(tickmode='array', tickvals=list(range(len(subset_features))), ticktext=subset_features),
            hovermode='x unified'
        )
        st.plotly_chart(fig)
    
    # User inputs for comparison
    st.header("Compare Actual and Predicted Votes")
    year = st.selectbox("Select Year", [2017, 2020, 2023])
    electorate = st.selectbox("Select Electorate", combined_result_list['Electorate'].unique())
    
    comparison_df = create_comparison_df(year, electorate)
    plot_comparison(comparison_df, year, electorate)
    
    # Function to plot 2024 predictions with color by party name as bar charts and add results number using Plotly
    party_colors = {
        'ACT New Zealand Vote': 'yellow',
        'Green Party Vote': 'green',
        'Labour Party Vote': 'red',
        'National Party Vote': 'blue',
        'New Zealand First Party Vote': 'black',
        'Others Vote': 'grey'
    }
    
    def plot_predictions_2024(predictions_df, electorate):
        predictions = predictions_df[predictions_df['Electorate'] == electorate]
        if predictions.empty:
            st.write(f"No data available for {electorate} in 2024.")
            return
    
        predictions = predictions.melt(id_vars=['Electorate'], value_vars=['ACT New Zealand Vote', 'Green Party Vote', 'Labour Party Vote', 'National Party Vote', 'New Zealand First Party Vote', 'Others Vote'], var_name='Party', value_name='Votes')
    
        fig = go.Figure()
        for party in predictions['Party'].unique():
            df_filtered = predictions[predictions['Party'] == party]
            fig.add_trace(go.Bar(x=[party], y=df_filtered['Votes'], name=party, marker_color=party_colors.get(party)))
    
        fig.update_layout(
            title=f'Predicted Votes for {electorate} in 2024',
            xaxis_title='Party',
            yaxis_title='Votes (%)',
            barmode='group',
            hovermode='x unified'
        )
        
        # Add vote counts on top of each bar
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    
        st.plotly_chart(fig)
    
    # User input for 2024 predictions
    st.header("Predictions for 2024")
    electorate_2024 = st.selectbox("Select Electorate for 2024", final_neural_predictions1_2024['Electorate'].unique())
    plot_predictions_2024(final_neural_predictions1_2024, electorate_2024)

# Display the selected page
if page == "Introduction":
    show_intro_page()
elif page == "Data Collection":
    show_Data_Collection_page()
elif page == "Regression Models":
    show_regression_page()
elif page == "KNN Regression":
    show_knn_page()
elif page == "Neural Network":
    show_nn_page()
