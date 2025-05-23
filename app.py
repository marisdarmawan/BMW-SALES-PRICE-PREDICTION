import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load the Model ---
try:
    with open('gradient_boosting_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file 'gradient_boosting_regression_model.pkl' not found. Please ensure it's in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Denormalization Function ---
# The notebook indicates that the 'price' was log-transformed using np.log().
# So, to convert back to the original scale, we use np.exp().
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

# Custom Input Layout based on actual training features
# Numerical inputs
st.header("Car Specifications")

col1, col2 = st.columns(2)
with col1:
    year = st.slider("Year of Manufacture", min_value=1996, max_value=2020, value=2017, help="Year the car was manufactured (based on historical data range).")
    mileage = st.number_input("Mileage (miles)", min_value=0, max_value=250000, value=25000, step=1000, help="Total miles driven by the car.")
    tax = st.number_input("Road Tax (GBP)", min_value=0, max_value=600, value=150, step=5, help="Annual road tax amount in British Pounds.")
with col2:
    mpg = st.number_input("MPG (Miles Per Gallon)", min_value=5.0, max_value=500.0, value=50.0, step=0.1, format="%.1f", help="Fuel efficiency of the car.")
    engine_size = st.number_input("Engine Size (liters)", min_value=0.0, max_value=7.0, value=2.0, step=0.1, format="%.1f", help="Engine size in liters (e.g., 2.0 for a 2-liter engine).")

st.markdown("---")
st.header("Categorical Features")

# Categorical inputs
# These options should ideally be derived from the unique categories present in the original training data.
# The lists below are comprehensive common options for BMW, and align with the one-hot encoding used in the notebook (pd.get_dummies).
# Users might need to adjust these lists if their training data has different unique values.

models = sorted(['1 Series', '2 Series', '3 Series', '4 Series', '5 Series', '6 Series', '7 Series', '8 Series', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z3', 'Z4', 'i3', 'i8', 'M2', 'M3', 'M4', 'M5', 'M6'])
model_selected = st.selectbox("Model", options=models, index=models.index('5 Series') if '5 Series' in models else 0)

transmissions = sorted(['Automatic', 'Manual', 'Semi-Auto'])
transmission_selected = st.selectbox("Transmission", options=transmissions)

fuel_types = sorted(['Diesel', 'Petrol', 'Hybrid', 'Electric', 'Other'])
fuel_type_selected = st.selectbox("Fuel Type", options=fuel_types)

st.markdown("---")

if st.button("Predict Price"):
    # Create a DataFrame from user inputs
    input_data = {
        'year': year,
        'mileage': mileage,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engine_size,
        'model': model_selected,
        'transmission': transmission_selected,
        'fuelType': fuel_type_selected
    }
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding similar to the training process
    # NOTE: It's crucial that the columns generated here match *exactly* those used during model training,
    # including the order and the categories dropped (due to drop_first=True).
    # In a production environment, you would save the list of training columns or the OneHotEncoder object.
    # For this demonstration, we'll infer the expected columns based on common practice with `drop_first=True`.

    # Define all possible categories to ensure consistent dummy column creation
    # (These are derived from the 'models', 'transmissions', 'fuel_types' lists defined above)
    all_models_for_dummies = sorted(models)
    all_transmissions_for_dummies = sorted(transmissions)
    all_fuel_types_for_dummies = sorted(fuel_types)

    # Create dummy variables for the input DataFrame
    # Use the full, sorted lists of categories to ensure all possible dummy columns are considered
    # and then align with the exact columns expected by the trained model.
    # This approach with `reindex` and `columns` ensures that if a category is missing in input,
    # its corresponding dummy column is set to 0.

    # Start with numerical columns
    processed_input = input_df[['year', 'mileage', 'tax', 'mpg', 'engineSize']].copy()

    # One-hot encode categorical features and append to processed_input
    # Ensure correct column names as expected by the model after drop_first=True
    # For 'model'
    model_dummies = pd.get_dummies(input_df['model'], prefix='model', drop_first=True)
    for col in sorted(all_models_for_dummies):
        if col != sorted(all_models_for_dummies)[0]: # Skip the dropped first category
            dummy_col_name = f'model_{col}'
            processed_input[dummy_col_name] = model_dummies[dummy_col_name] if dummy_col_name in model_dummies.columns else 0

    # For 'transmission'
    transmission_dummies = pd.get_dummies(input_df['transmission'], prefix='transmission', drop_first=True)
    for col in sorted(all_transmissions_for_dummies):
        if col != sorted(all_transmissions_for_dummies)[0]: # Skip the dropped first category
            dummy_col_name = f'transmission_{col}'
            processed_input[dummy_col_name] = transmission_dummies[dummy_col_name] if dummy_col_name in transmission_dummies.columns else 0

    # For 'fuelType'
    fuel_type_dummies = pd.get_dummies(input_df['fuelType'], prefix='fuelType', drop_first=True)
    for col in sorted(all_fuel_types_for_dummies):
        if col != sorted(all_fuel_types_for_dummies)[0]: # Skip the dropped first category
            dummy_col_name = f'fuelType_{col}'
            processed_input[dummy_col_name] = fuel_type_dummies[dummy_col_name] if dummy_col_name in fuel_type_dummies.columns else 0

    # Define the exact list of columns that the model was trained on.
    # This list must be obtained from the training script.
    # The order of columns is also critical.
    # Based on the notebook's use of pd.get_dummies(drop_first=True), and standard practice
    # of numerical columns followed by one-hot encoded columns.
    # This is an inferred list and *must be verified against the actual training feature set*.

    # Numerical features
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

    # Generate expected dummy columns dynamically based on the assumed `drop_first` logic
    expected_model_dummies = [f'model_{m}' for m in sorted(all_models_for_dummies) if m != sorted(all_models_for_dummies)[0]]
    expected_transmission_dummies = [f'transmission_{t}' for t in sorted(all_transmissions_for_dummies) if t != sorted(all_transmissions_for_dummies)[0]]
    expected_fuel_type_dummies = [f'fuelType_{f}' for f in sorted(all_fuel_types_for_dummies) if f != sorted(all_fuel_types_for_dummies)[0]]

    expected_training_columns = numerical_cols + expected_model_dummies + expected_transmission_dummies + expected_fuel_type_dummies

    # Reindex `processed_input` to match the `expected_training_columns`
    # This ensures consistency in column order and presence (filling missing with 0)
    final_input_df = processed_input.reindex(columns=expected_training_columns, fill_value=0)

    try:
        # Make prediction
        predicted_log_price = model.predict(final_input_df)[0]

        # Denormalize the predicted price
        predicted_original_price = denormalize_price(predicted_log_price)

        st.success(f"Estimated BMW Used Car Price: **£{predicted_original_price:,.2f}**")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
    ---
    **Note on Feature Engineering**:
    The model was trained on data where 'price' was log-transformed and categorical features
    ('model', 'transmission', 'fuelType') were one-hot encoded with `drop_first=True`.
    The input processing in this app mirrors these transformations. For robust deployment,
    it is highly recommended to save the exact column names and their order from your training dataset
    (e.g., `X_train.columns.tolist()`) and load them here to align new inputs precisely.
""")
