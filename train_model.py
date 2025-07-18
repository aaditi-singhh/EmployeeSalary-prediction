import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and prepare your dataset
df = pd.read_csv("adult 3.csv")

# Lowercase all column names
df.columns = df.columns.str.lower()

# Handle missing values
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Drop unnecessary columns
if 'fnlwgt' in df.columns:
    df.drop(['fnlwgt'], axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Save encoders
joblib.dump(label_encoders, 'encoders.pkl')

# Feature/Target split
x = df.drop("income", axis=1)
y = df["income"]

# Scale features
scaler = StandardScaler()
x = scaler.fit_transform(x)
joblib.dump(scaler, "scaler.pkl")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("✅ Saved best model as best_model.pkl")
