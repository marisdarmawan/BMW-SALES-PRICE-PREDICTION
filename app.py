import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the Model ---
try:
    model = joblib.load('gradient_boosting_regression_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'gradient_boosting_regression_model.pkl' not found. Please ensure it's in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Denormalization Function ---
def denormalize_price(log_price):
    return np.exp(log_price)

# --- Streamlit App Layout ---
st.set_page_config(page_title="BMW Used Car Price Predictor", layout="centered")

st.title("🚗 BMW Used Car Price Prediction")
st.markdown("---")

st.write("""
    This app estimates the selling price of a used BMW car based on its key features.
    Please enter the car details below:
""")

# Numerical inputs
st.header("Car Specifications")

col1, col2 = st.columns(2)
with col1:
    year = st.slider("Year of Manufacture", 1996, 2020, 2017)
    mileage = st.number_input("Mileage (miles)", 0, 250000, 25000, 1000)
    tax = st.number_input("Road Tax (GBP)", 0, 600, 150, 5)
with col2:
    mpg = st.number_input("MPG", 5.0, 500.0, 50.0, 0.1, format="%.1f")
    engine_size = st.number_input("Engine Size (liters)", 0.0, 7.0, 2.0, 0.1, format="%.1f")

st.markdown("---")
st.header("Categorical Features")

models = sorted(['1 Series', '2 Series', '3 Series', '4 Series', '5 Series', '6 Series', '7 Series', '8 Series', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z3', 'Z4', 'i3', 'i8', 'M2', 'M3', 'M4', 'M5', 'M6'])
transmissions = sorted(['Automatic', 'Manual', 'Semi-Auto'])
fuel_types = sorted(['Diesel', 'Petrol', 'Hybrid', 'Electric', 'Other'])

model_selected = st.selectbox("Model", options=models, index=models.index('5 Series') if '5 Series' in models else 0)
transmission_selected = st.selectbox("Transmission", options=transmissions)
fuel_type_selected = st.selectbox("Fuel Type", options=fuel_types)

st.markdown("---")

if st.button("Predict Price"):
    input_data = {
        'year': year,
        'mileage': mileage,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engine_size,
        'transmission': transmission_selected,
        'fuelType': fuel_type_selected,
        'series': model_selected.split()[0]  # extract series from model
    }

    # Extract 'type' from model
    def get_type(model_name):
        if 'Series' in model_name:
            return 'S'
        elif 'X' in model_name:
            return 'X'
        elif 'i' in model_name:
            return 'i'
        elif 'Z' in model_name:
            return 'Z'
        return 'Other'

    input_data['type'] = get_type(model_selected)
    input_df = pd.DataFrame([input_data])

    # Start with numerical features
    processed_input = input_df[['year', 'mileage', 'tax', 'mpg', 'engineSize']].copy()

    # Generate one-hot encoded columns without prefix
    trans_dummies = pd.get_dummies(input_df['transmission'], prefix='', prefix_sep='', drop_first=True)
    fuel_dummies = pd.get_dummies(input_df['fuelType'], prefix='', prefix_sep='', drop_first=True)
    type_dummies = pd.get_dummies(input_df['type'], prefix='', prefix_sep='', drop_first=True)

    # Combine all features
    processed_input = pd.concat([processed_input, trans_dummies, fuel_dummies, type_dummies], axis=1)

    # Add 'Electric' column if fuel type is electric
    if fuel_type_selected == 'Electric':
        processed_input['Electric'] = 1
    else:
        processed_input['Electric'] = 0

    # Add 'series' column
    processed_input['series'] = input_data['series']

    # Match columns expected by model
    expected_columns = [
        'year', 'mileage', 'tax', 'mpg', 'engineSize',
        'Manual', 'Semi-Auto',
        'Electric', 'Hybrid', 'Other', 'Petrol',
        'S', 'X', 'Z', 'i',
        'series'
    ]
    final_input_df = processed_input.reindex(columns=expected_columns, fill_value=0)

    # Predict
    try:
        predicted_log_price = model.predict(final_input_df)[0]
        predicted_original_price = denormalize_price(predicted_log_price)
        st.success(f"Estimated BMW Used Car Price: **£{predicted_original_price:,.2f}**")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
