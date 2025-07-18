import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("knn_salary_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("💼 Employee Salary Prediction")

with st.form("prediction_form"):
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", list(encoders['workclass'].keys()))
    education = st.selectbox("Education", list(encoders['education'].keys()))
    marital_status = st.selectbox("Marital Status", list(encoders['marital-status'].keys()))
    occupation = st.selectbox("Occupation", list(encoders['occupation'].keys()))
    relationship = st.selectbox("Relationship", list(encoders['relationship'].keys()))
    race = st.selectbox("Race", list(encoders['race'].keys()))
    sex = st.radio("Sex", list(encoders['sex'].keys()))
    capital_gain = st.number_input("Capital Gain", 0)
    capital_loss = st.number_input("Capital Loss", 0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", list(encoders['native-country'].keys()))
    
    submit = st.form_submit_button("Predict")  # ✅ REQUIRED

if submit:
    df = pd.DataFrame([[
        age,
        encoders['workclass'][workclass],
        encoders['education'][education],
        encoders['marital-status'][marital_status],
        encoders['occupation'][occupation],
        encoders['relationship'][relationship],
        encoders['race'][race],
        encoders['sex'][sex],
        capital_gain,
        capital_loss,
        hours_per_week,
        encoders['native-country'][native_country]
    ]])

    df_scaled = scaler.transform(df)
    result = model.predict(df_scaled)[0]
    st.success(f"🎯 Predicted Salary: {'>50K' if result == 1 else '<=50K'}")
