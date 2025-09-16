# streamlit_app.py
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import requests

# Paths
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

# Load model + scaler
model = joblib.load(MODEL_DIR / "mitbih_1_rf.pkl")
scaler = joblib.load(MODEL_DIR / "mitbih_1_scaler.pkl")

# Streamlit config
st.set_page_config(page_title="Heart/ECG Risk Detector", layout="wide")
st.title("❤️ Heart / ECG Risk Detector")

# Sidebar
st.sidebar.header("Settings")
use_api = st.sidebar.checkbox("Use Flask API", value=False)
api_url = st.sidebar.text_input("Flask API URL", value="http://127.0.0.1:5000/predict")
mode = st.sidebar.radio("Input mode", ["Paste single row", "Upload CSV (batch)", "Use sample row"])

# Helper
def risk_message(pred):
    if pred == 0:
        return "✅ No significant risk detected."
    else:
        return "⚠️ High risk detected! Please consult a doctor immediately."

# --- Single row input ---
if mode == "Paste single row":
    txt = st.text_area("Paste 187 comma-separated features (no label):")
    if st.button("Predict"):
        try:
            features = [float(x.strip()) for x in txt.split(",") if x.strip() != ""]
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]
            probs = model.predict_proba(features_scaled)[0]

            st.success(f"Prediction: Class {pred}")
            st.info(f"Confidence: {probs.max()*100:.2f}%")
            st.warning(risk_message(pred))

            fig, ax = plt.subplots()
            ax.bar(range(len(probs)), probs)
            ax.set_xlabel("Class")
            ax.set_ylabel("Probability")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# --- Batch upload ---
elif mode == "Upload CSV (batch)":
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up, header=None)
        X = df.iloc[:, :-1].astype(float).values if df.shape[1] == scaler.n_features_in_ + 1 else df.astype(float).values
        Xs = scaler.transform(X)
        preds = model.predict(Xs)
        probs = model.predict_proba(Xs).max(axis=1)

        out = pd.DataFrame({"pred": preds, "confidence": probs})
        st.dataframe(out.head(30))

        csv = out.to_csv(index=False).encode()
        st.download_button("Download predictions", csv, "predictions.csv")

# --- Sample row ---
elif mode == "Use sample row":
    csv_path = ROOT / "Datasets" / "mitbih_test.csv"
    if not csv_path.exists():
        st.warning("⚠️ No sample dataset found in Datasets/")
    else:
        df = pd.read_csv(csv_path, header=None)
        sr = df.iloc[0, :-1].astype(float).tolist()
        st.write(f"Loaded sample row with {len(sr)} features.")

        if st.button("Predict sample"):
            Xs = scaler.transform([sr])
            pred = model.predict(Xs)[0]
            probs = model.predict_proba(Xs)[0]

            st.success(f"Prediction: Class {pred}")
            st.info(f"Confidence: {probs.max()*100:.2f}%")
            st.warning(risk_message(pred))

            fig, ax = plt.subplots()
            ax.plot(sr)
            ax.set_title("ECG Beat (Sample)")
            st.pyplot(fig)

# --- Dataset EDA ---
st.markdown("---")
st.header("📊 Quick EDA (Dataset)")
csv_path = ROOT / "Datasets" / "mitbih_test.csv"
if csv_path.exists():
    df_all = pd.read_csv(csv_path, header=None)
    st.write("Dataset shape:", df_all.shape)
    labels = df_all.iloc[:, -1].astype(int)
    st.bar_chart(labels.value_counts())
else:
    st.info("Upload mitbih_test.csv into Datasets/ folder.")
