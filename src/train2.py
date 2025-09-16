# src/train2.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
train_path = "../Datasets/mitbih_train.csv"
df = pd.read_csv(train_path, header=None)

# Features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "../models/model_1_rf.pkl")
joblib.dump(scaler, "../models/mode_1_scaler.pkl")

print("Training completed. Model and scaler saved in 'models/' folder.")
