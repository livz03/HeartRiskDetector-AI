# train2.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Datasets" / "mitbih_train.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, header=None)
X = df.iloc[:, :-1].astype(float).values
y = df.iloc[:, -1].astype(int).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

<<<<<<< HEAD
print(" Model and scaler saved: models/mitbih_1_rf.pkl & models/mitbih_1_scaler.pkl")
=======
# Save artifacts
joblib.dump(model, MODEL_DIR / "mitbih_1_rf.pkl")
joblib.dump(scaler, MODEL_DIR / "mitbih_1_scaler.pkl")

print(f"✅ Model and scaler saved in {MODEL_DIR}")
>>>>>>> 21ac45e (again replaced the scripts of train2 and streamlit_run)
