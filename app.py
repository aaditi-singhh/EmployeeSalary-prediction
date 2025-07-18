import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing tools
# IMPORTANT: Ensure these .pkl files (best_model.pkl, scaler.pkl, encoders.pkl)
# are in the same directory as your app.py file, or provide their full paths.
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model or encoders: {e}. "
             "Please ensure 'best_model.pkl', 'scaler.pkl', and 'encoders.pkl' "
             "are in the same directory as this Streamlit script.")
    st.stop() # Stop the app execution if essential files are missing

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Enter employee details below to predict if salary is **>50K** or **<=50K**.")

# --- Streamlit Form for User Input ---
with st.form("prediction_form"):
    st.subheader("Employee Details:")

    age = st.slider("Age", 17, 90, 30)
    
    # Selectbox for categorical features, using keys from encoders
    workclass = st.selectbox("Workclass", list(encoders['workclass'].keys()))
    education = st.selectbox("Education", list(encoders['education'].keys()))
    
    # Numeric input for education_num (which corresponds to 'education-num' in the model)
    education_num = st.slider("Education Number (Years of Education)", 1, 16, 10)
    
    marital_status = st.selectbox("Marital Status", list(encoders['marital-status'].keys()))
    occupation = st.selectbox("Occupation", list(encoders['occupation'].keys()))
    relationship = st.selectbox("Relationship", list(encoders['relationship'].keys()))
    race = st.selectbox("Race", list(encoders['race'].keys()))
    
    # Radio button for Sex (using 'sex' lowercase key)
    sex = st.radio("Sex", list(encoders['sex'].keys()))
    
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0, help="e.g., 0 for no gain, up to 99999")
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0, help="e.g., 0 for no loss, up to 4356")
    
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", list(encoders['native-country'].keys()))

    # --- Submit Button ---
    submit = st.form_submit_button("Predict Salary")

# --- Prediction Logic ---
if submit:
    try:
        # Create a list of input values in the exact order expected by the model
        # Ensure all categorical features are mapped to their encoded numerical values
        input_data_list = [[
            age,
            encoders['workclass'][workclass],
            encoders['education'][education],
            education_num, # Numeric value from slider
            encoders['marital-status'][marital_status],
            encoders['occupation'][occupation],
            encoders['relationship'][relationship],
            encoders['race'][race],
            encoders['sex'][sex], # Using 'sex' (lowercase)
            capital_gain,
            capital_loss,
            hours_per_week,
            encoders['native-country'][native_country]
        ]]

        # Create a Pandas DataFrame from the input list
        # IMPORTANT: Column names must exactly match the features the scaler/model were trained on,
        # including case and hyphens/underscores.
        input_df = pd.DataFrame(input_data_list, columns=[
            'age', 'workclass', 'education', 'education-num', 'marital-status', # 'education-num' with hyphen
            'occupation', 'relationship', 'race', 'sex', 'capital-gain',
            'capital_loss', 'hours-per-week', 'native-country' # 'hours-per-week' with hyphen (from training script)
        ])
        
        # Note: The training script shows 'hours_per_week' in the slider, but 'hours-per-week' in the column list
        # Based on typical adult dataset preprocessing, 'hours-per-week' (hyphen) is com
