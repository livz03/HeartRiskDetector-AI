# streamlit_app.py
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import requests

from src.predict2 import predict_row, _load

st.set_page_config(page_title="Heart Risk Detector", layout="wide")
ROOT = Path(__file__).resolve().parent

st.title("❤️ Heart / ECG Risk Detector")

# Sidebar
st.sidebar.header("Settings")
use_api = st.sidebar.checkbox("Use Flask API for predictions", value=False)
api_url = st.sidebar.text_input("Flask API URL", value="http://127.0.0.1:5000/predict")

mode = st.sidebar.radio("Input mode", ["Paste single row", "Upload CSV (batch)", "Use sample row"])

# --- Helpers ---
def sample_row():
    p = ROOT / "Datasets" / "mitbih_test.csv"
    if p.exists():
        df = pd.read_csv(p, header=None)
        return df.iloc[0, :-1].astype(float).tolist()
    return None

def call_api(features):
    """Send features to Flask API"""
    try:
        r = requests.post(api_url, json={"features": features}, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"API returned {r.status_code}: {r.text}"}
    except Exception as e:
        return {"error": str(e)}

# --- Modes ---
if mode == "Paste single row":
    txt = st.text_area("Paste 187 comma-separated features (no label):")
    if st.button("Predict"):
        try:
            features = [float(x.strip()) for x in txt.split(",") if x.strip()]
            out = call_api(features) if use_api else predict_row(features)

            if "error" in out:
                st.error(out["error"])
            else:
                st.success(f"Prediction: {out['label']} (class {out['class']})")
                st.info(f"Confidence: {out['confidence']*100:.2f}%")
                fig, ax = plt.subplots()
                ax.bar(range(len(out["probs"])), out["probs"])
                ax.set_xlabel("Class")
                ax.set_ylabel("Probability")
                st.pyplot(fig)
        except Exception as e:
            st.error(str(e))

elif mode == "Upload CSV (batch)":
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up, header=None)
        st.write("Loaded:", df.shape)

        try:
            scaler = joblib.load(ROOT / "models" / "mitbih_1_scaler.pkl")
            expected = scaler.n_features_in_
        except:
            expected = None

        if expected and df.shape[1] == expected + 1:
            X = df.iloc[:, :-1].astype(float).values
            y = df.iloc[:, -1].astype(int).values
        else:
            X = df.astype(float).values
            y = None

        if use_api:
            results = []
            for i, row in df.iterrows():
                features = row[:-1].tolist() if expected and df.shape[1] == expected+1 else row.tolist()
                results.append(call_api(features))
            st.json(results[:5])
        else:
            model, scaler = _load()
            Xs = scaler.transform(X)
            preds = model.predict(Xs)
            probs = model.predict_proba(Xs).max(axis=1)
            out = pd.DataFrame({"pred": preds, "confidence": probs})
            st.dataframe(out.head(20))
            csv = out.to_csv(index=False).encode()
            st.download_button("Download predictions", csv, "predictions.csv")

elif mode == "Use sample row":
    sr = sample_row()
    if sr is None:
        st.warning("No sample found in Datasets/mitbih_test.csv")
    else:
        st.write(f"Loaded sample row with {len(sr)} values.")
        if st.button("Predict sample"):
            out = call_api(sr) if use_api else predict_row(sr)
            if "error" in out:
                st.error(out["error"])
            else:
                st.success(f"Prediction: {out['label']} (class {out['class']})")
                st.info(f"Confidence: {out['confidence']*100:.2f}%")
                fig, ax = plt.subplots()
                ax.plot(sr)
                ax.set_title("ECG Beat (sample)")
                st.pyplot(fig)

# --- Quick Dataset EDA ---
st.markdown("---")
st.header("Quick Dataset EDA")
csv_path = ROOT / "Datasets" / "mitbih_test.csv"
if csv_path.exists():
    df_all = pd.read_csv(csv_path, header=None)
    st.write("Dataset shape:", df_all.shape)
    st.write("Class distribution:")
    labels = df_all.iloc[:, -1].astype(int)
    st.bar_chart(labels.value_counts())
else:
    st.info("Put mitbih_test.csv into Datasets/ and refresh.")
