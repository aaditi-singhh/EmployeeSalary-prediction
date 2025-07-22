# 💼 Employee Salary Prediction App

This project predicts whether an employee's salary is **>50K or <=50K** using machine learning models trained on the **Adult Income Dataset**. It includes model training, data preprocessing, and a Streamlit web app for user interaction.

---

## 📁 Project Structure

```
employee-salary-prediction/
│
├── app.py                  # Streamlit app for prediction
├── train_model.py          # Model training & preprocessing script
├── adult 3.csv             # Dataset used
├── best_model.pkl          # Best performing ML model (saved)
├── scaler.pkl              # StandardScaler for numeric features
├── encoders.pkl            # Encoders for categorical features
├── requirements.txt        # Required Python libraries
└── README.md               # You're here!
```

---

## 👩‍💻 Built By
**Aditi Singh**  
ITER, SOA University

---

## 🧠 Model Training

- Categorical features are encoded using `LabelEncoder`.
- Features are standardized using `StandardScaler`.
- Multiple models were tested:
  - Logistic Regression
  - Random Forest
  - KNN
  - SVM
  - Gradient Boosting
- The **best performing model** is saved as `best_model.pkl`.

### Training Highlights:
- Dataset cleaned (missing values removed)
- Accuracy printed for all models
- Best model auto-selected and saved

---

## 🚀 Streamlit Web App Features

Users can input the following details:

- Age
- Final Weight (fnlwgt)
- Workclass
- Education
- Education Number (Years)
- Marital Status
- Occupation
- Relationship
- Race
- Capital Gain / Loss
- Hours per Week
- Native Country

It returns a prediction:

> ✅ **Predicted Salary: >50K or <=50K**

---

## 🛠️ How to Run Locally

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

2. **Run the app**
```bash
streamlit run app.py  
```

OR 

pip install --force-reinstall pandas scikit-learn pyarrow

Run in terminal

## 🌐 How to Deploy on Streamlit Cloud

1. Push the project to a GitHub repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub → select repo → deploy
4. App will run at:  
   `https://<your-username>-<repo-name>.streamlit.app/`

---

## 📦 Requirements

Make sure `requirements.txt` contains:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 📊 Dataset Reference

- UCI Adult Income Dataset  
- Contains demographic data to predict if income exceeds $50K/year  
- Source: [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

---

## 📌 Notes

- All encoders use lowercase column names (`df.columns.str.lower()`).
- The app currently logs encoder keys with `st.write()` for debugging. You may remove this before deployment.

---

## 📬 Contact

If you liked this project or want to collaborate, feel free to connect with **Aditi Singh**.
