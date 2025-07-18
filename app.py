import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing tools
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Salary Predictor")
st.title("💼 Employee Salary Prediction App")

with st.form("prediction_form"):
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", list(encoders['workclass'].keys()))
    education = st.selectbox("Education", list(encoders['education'].keys()))
    education_num = st.slider("Education Number (Years of Education)", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", list(encoders['marital-status'].keys()))
    occupation = st.selectbox("Occupation", list(encoders['occupation'].keys()))
    relationship = st.selectbox("Relationship", list(encoders['relationship'].keys()))
    race = st.selectbox("Race", list(encoders['race'].keys()))
 

    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", list(encoders['native-country'].keys()))
    
    submit = st.form_submit_button("Predict Salary")
print(X.columns.tolist()) 
if submit:
    try:
       input_data = [[
    age,
 
    encoders['workclass'][workclass],
    encoders['education'][education],
    education_num,
    encoders['marital-status'][marital_status],
    encoders['occupation'][occupation],
    encoders['relationship'][relationship],
    encoders['race'][race],
    encoders['sex'][sex],
    capital_gain,
    capital_loss,
    hours_per_week,
    encoders['native-country'][native_country]
]]

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"🎯 Predicted Salary: **{result}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
