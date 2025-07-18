import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing tools
# Ensure these .pkl files are in the same directory as your app.py or provide full paths
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model or encoders: {e}. Make sure 'best_model.pkl', 'scaler.pkl', and 'encoders.pkl' are in the correct directory.")
    st.stop() # Stop the app if essential files are missing

st.set_page_config(page_title="Employee Salary Predictor", layout="centered") # Added layout for better centering
st.title("💼 Employee Salary Prediction App")
st.markdown("Enter employee details below to predict if salary is **>50K** or **<=50K**.")

with st.form("prediction_form"):
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", list(encoders['workclass'].keys()))
    education = st.selectbox("Education", list(encoders['education'].keys()))
    education-num = st.slider("Education Number (Years of Education)", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", list(encoders['marital-status'].keys()))
    occupation = st.selectbox("Occupation", list(encoders['occupation'].keys()))
    relationship = st.selectbox("Relationship", list(encoders['relationship'].keys()))
    race = st.selectbox("Race", list(encoders['race'].keys()))
   
   
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", list(encoders['native-country'].keys()))

    submit = st.form_submit_button("Predict Salary")

if submit:
    try:
        input_data = [[
            age,
            encoders['workclass'][workclass],
            encoders['education'][education],
            education-num,
            encoders['marital-status'][marital_status],
            encoders['occupation'][occupation],
            encoders['relationship'][relationship],
            encoders['race'][race],
            # Corrected 'Sex' to 'sex' (lowercase 's') here
          
            capital_gain,
            capital_loss,
            hours_per_week,
            encoders['native-country'][native_country]
        ]]

        # Ensure input_data is a DataFrame or NumPy array if scaler expects it
        # Pandas DataFrame is often safer for scikit-learn pipelines
        input_df = pd.DataFrame(input_data, columns=[
            'age', 'workclass', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race',  'capital_gain',
            'capital_loss', 'hours_per_week', 'native-country'
        ])

        input_scaled = scaler.transform(input_df) # Use input_df here
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"🎯 Predicted Salary: **{result}**")
    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
        st.info("Please check the input values and ensure the loaded model and encoders are compatible.")

