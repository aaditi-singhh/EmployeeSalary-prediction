import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("knn_salary_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Enter employee details below to predict if the salary is **>50K** or **<=50K**.")

# 📝 Input Form
with st.form("prediction_form"):
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", list(encoders['workclass'].keys()))
    education = st.selectbox("Education", list(encoders['education'].keys()))
    marital_status = st.selectbox("Marital Status", list(encoders['marital-status'].keys()))
    occupation = st.selectbox("Occupation", list(encoders['occupation'].keys()))
    relationship = st.selectbox("Relationship", list(encoders['relationship'].keys()))
    race = st.selectbox("Race", list(encoders['race'].keys()))
    sex = st.radio("Sex", list(encoders['sex'].keys()))
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", list(encoders['native-country'].keys()))

    # ✅ The required Submit Button (must be inside the form block)
    submit = st.form_submit_button("Predict Salary")

# 🧠 Predict when form is submitted
if submit:
    try:
        input_data = pd.DataFrame([[
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

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"

        st.success(f"🎯 Predicted Salary: **{result}**")

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
