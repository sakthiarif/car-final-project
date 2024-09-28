import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load models and encoders
linear_model = joblib.load('linear_regression_model.pkl')
svr_model = joblib.load('svr_model.pkl')
knn_model = joblib.load('knn_model.pkl')
pls_model = joblib.load('pls_model.pkl')
gbr_model = joblib.load('gbr_model.pkl')
mlp_model = joblib.load('mlp_model.pkl')
encoder = joblib.load('encoder.pkl')  # TargetEncoder or OneHotEncoder
scaler = joblib.load('scaler.pkl')      # MinMaxScaler

# Function to get user input
def get_user_input():
    make = st.selectbox("Make", ["Toyota", "Honda", "Ford", "Chevrolet", "Nissan"])
    year = st.number_input("Year", min_value=2000, max_value=2023, value=2021)
    mileage = st.number_input("Mileage (in miles)", min_value=0, value=50000)
    engine_hp = st.number_input("Engine HP", min_value=0, value=150)
    transmission_type = st.selectbox("Transmission Type", ["Automatic", "Manual"])
    vehicle_size = st.selectbox("Vehicle Size", ["Compact", "Midsize", "Fullsize"])

    input_data = {
        'Make': make,
        'Year': year,
        'Mileage': mileage,
        'Engine HP': engine_hp,
        'Transmission Type': transmission_type,
        'Vehicle Size': vehicle_size,
    }
    return pd.DataFrame(input_data, index=[0])

# Streamlit UI
st.title("Car Price Prediction")
st.write("Enter the details of the car below:")

user_input = get_user_input()

if st.button("Predict Price"):
    # Transform input data
    encoded_input = encoder.transform(user_input[['Make', 'Transmission Type', 'Vehicle Size']])
    scaled_input = scaler.transform(user_input[['Year', 'Mileage', 'Engine HP']])
    input_data = np.hstack((encoded_input, scaled_input))

    # Make predictions with different models
    predictions = {
        'Linear Regression': linear_model.predict(input_data),
        'SVR': svr_model.predict(input_data),
        'KNN': knn_model.predict(input_data),
        'PLS Regression': pls_model.predict(input_data),
        'Gradient Boosting': gbr_model.predict(input_data),
        'MLP': mlp_model.predict(input_data)
    }
    
    # Display predictions
    for model_name, prediction in predictions.items():
        st.write(f"{model_name}: ${prediction[0]:,.2f}")

# Optional: Add plots or insights based on your previous analysis
st.subheader("Data Insights")
# Example: Show a heatmap or other plots if required
