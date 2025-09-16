import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Load dataset (replace path if needed)
# -----------------------------
df = pd.read_csv("datasets/heart.csv")

# Example dataset should include columns like:
# Age, Sex, Cholesterol, RestingBP, ECG, HeartRate, target

# Encode categorical values
df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "models/model_1_rf.pkl")
joblib.dump(scaler, "models/model_1_scaler.pkl")

print(" Model and scaler saved: models/mitbih_1_rf.pkl & models/mitbih_1_scaler.pkl")
