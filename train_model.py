import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv("adult 3.csv")

# Lowercase all column names (to avoid KeyErrors in app)
df.columns = df.columns.str.lower()

# Clean missing data
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Drop unnecessary column if present
if 'fnlwgt' in df.columns:
    df.drop('fnlwgt', axis=1, inplace=True)

# Label encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Save encoders
joblib.dump(label_encoders, 'encoders.pkl')

# Split features and target
X = df.drop("income", axis=1)
y = df["income"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'knn_salary_model.pkl')
print("✅ Model training complete and saved!")
