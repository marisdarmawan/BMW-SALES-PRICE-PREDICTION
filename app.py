import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the Model ---
try:
    model = joblib.load('gradient_boosting_regression_model.pkl')
    power_transformer = joblib.load('price_transformer.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'gradient_boosting_regression_model.pkl' not found. Please ensure it's in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Denormalization Function ---
def denormalize_price(transformed_price, transformer):
    # The inverse_transform method expects a 2D array
    return transformer.inverse_transform([[transformed_price]])[0][0]

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
    # Function to extract series (should mirror your notebook's logic)
    def get_series_notebook(model_name_str):
        if '1 Series' in model_name_str or (not 'Series' in model_name_str and '1' in model_name_str and not model_name_str.startswith('X1')and not model_name_str.startswith('i')): return 1
        elif '2 Series' in model_name_str or (not 'Series' in model_name_str and '2' in model_name_str): return 2
        elif '3 Series' in model_name_str or (not 'Series' in model_name_str and '3' in model_name_str and not model_name_str.startswith('i')): return 3
        elif '4 Series' in model_name_str or (not 'Series' in model_name_str and '4' in model_name_str): return 4
        elif '5 Series' in model_name_str or (not 'Series' in model_name_str and '5' in model_name_str): return 5
        elif '6 Series' in model_name_str or (not 'Series' in model_name_str and '6' in model_name_str): return 6
        elif '7 Series' in model_name_str or (not 'Series' in model_name_str and '7' in model_name_str): return 7
        elif '8 Series' in model_name_str or (not 'Series' in model_name_str and '8' in model_name_str and not model_name_str.startswith('i')): return 8
        # Handling for X series models like X1, X2 etc. if they are treated as series 1, 2...
        # This part needs to be exactly how your notebook's get_series handles them.
        # The notebook's get_series just checks for '1' in string, '2' in string etc.
        # So "M4" would return 4. "Z3" would return 3. "i3" would return 3. "X5" would return 5.
        # If a model like "M" or "X" (without a number) was meant to be series 0 in the notebook:
        if 'M' == model_name_str.strip() : return 0 # Example for plain "M"
        if 'X' == model_name_str.strip() : return 0 # Example for plain "X"

        # Default for models where series number is not explicitly 1-8 or extracted.
        # This should match how NaNs from get_series were handled in the notebook (e.g., filled with 0).
        return 0

    # Function to extract type (should mirror your notebook's logic)
    def get_type_notebook(model_name_str):
        if 'Series' in model_name_str: return 'S'
        elif 'X' in model_name_str: return 'X'
        elif 'i' in model_name_str: return 'i'
        elif 'Z' in model_name_str: return 'Z'
        elif 'M' in model_name_str: return 'M'
        return 'Unknown' # Fallback, though all models in your list should be covered

    # Prepare a dictionary for the single row of input
    input_data_dict = {
        'year': year,
        'mileage': mileage,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engine_size,
        'Manual': 0, 'Semi-Auto': 0,
        'Electric': 0, 'Hybrid': 0, 'Other': 0, 'Petrol': 0, # 'Other' for fuelType
        'series': get_series_notebook(model_selected),
        'S': 0, 'X': 0, 'Z': 0, 'i': 0 # 'i' for car type
    }

    # Set transmission dummies (Automatic is the base category if drop_first=True was used)
    if transmission_selected == 'Manual':
        input_data_dict['Manual'] = 1
    elif transmission_selected == 'Semi-Auto':
        input_data_dict['Semi-Auto'] = 1

    # Set fuelType dummies (Diesel is the base category if drop_first=True was used)
    if fuel_type_selected == 'Electric':
        input_data_dict['Electric'] = 1
    elif fuel_type_selected == 'Hybrid':
        input_data_dict['Hybrid'] = 1
    elif fuel_type_selected == 'Other': # This is for fuelType 'Other'
        input_data_dict['Other'] = 1
    elif fuel_type_selected == 'Petrol':
        input_data_dict['Petrol'] = 1

    # Set type dummies (M type is the base category if drop_first=True was used)
    car_type = get_type_notebook(model_selected)
    if car_type == 'S':
        input_data_dict['S'] = 1
    elif car_type == 'X':
        input_data_dict['X'] = 1
    elif car_type == 'Z':
        input_data_dict['Z'] = 1
    elif car_type == 'i': # This is for car type 'i'
        input_data_dict['i'] = 1

    # Define the exact column order as expected by the trained model
    # This list should precisely match the columns of X_train in your notebook
    expected_columns_from_notebook = [
        'year', 'mileage', 'tax', 'mpg', 'engineSize',
        'Manual', 'Semi-Auto',
        'Electric', 'Hybrid', 'Other', 'Petrol', # 'Other' for fuelType
        'series',
        'S', 'X', 'Z', 'i' # 'i' for car type
    ]

    # Create a single-row DataFrame with all expected columns initialized
    final_input_df_data = {col: [0] for col in expected_columns_from_notebook} # Initialize with a list for single row
    for key, value in input_data_dict.items():
        if key in final_input_df_data:
            final_input_df_data[key] = [value] # Ensure value is in a list for single row DataFrame

    final_input_df = pd.DataFrame(final_input_df_data, columns=expected_columns_from_notebook)

    # Ensure 'series' is numeric (it should be from get_series_notebook)
    final_input_df['series'] = pd.to_numeric(final_input_df['series'])

    # Predict
    try:
        # For debugging, you can uncomment these lines in your app.py:
        # st.write("--- Debug Info ---")
        # st.write("Selected Model:", model_selected)
        # st.write("Derived Series:", input_data_dict['series'])
        # st.write("Derived Type:", car_type)
        # st.write("Input Data to Model (first row):")
        # st.dataframe(final_input_df.head(1))
        # st.write("Data Types of Input DataFrame:")
        # st.write(final_input_df.dtypes)

        predicted_transformed_price = model.predict(final_input_df)[0]
        predicted_original_price = denormalize_price(predicted_transformed_price, power_transformer)
        st.success(f"Estimated BMW Used Car Price: **£{predicted_original_price:,.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please double-check the input values and ensure the model files are correctly loaded and compatible.")
        # st.dataframe(final_input_df) # Show the problematic dataframe for debugging
        # st.write(final_input_df.dtypes)
