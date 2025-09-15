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
use_api = st.sidebar.checkbox("Route predictions via Flask API", value=False)
api_url = st.sidebar.text_input("Flask API URL", value="http://127.0.0.1:5000/predict")

mode = st.sidebar.radio("Input mode", ["Paste single row", "Upload CSV (batch)", "Use sample row"])

# Load a sample row
def sample_row():
    p = ROOT / "Datasets" / "mitbih_test.csv"
    if p.exists():
        df = pd.read_csv(p, header=None)
        return df.iloc[0, :-1].astype(float).tolist()
    return None

# ---- Paste Single Row ----
if mode == "Paste single row":
    txt = st.text_area("Paste 187 comma-separated features (no label):")
    if st.button("Predict"):
        try:
            features = [float(x.strip()) for x in txt.split(",") if x.strip()]
            if use_api:
                r = requests.post(api_url, json={"features": features}, timeout=10)
                st.write(r.json())
            else:
                out = predict_row(features)
                st.success(f"Prediction: {out['label']} (class {out['class']})")
                st.info(f"Confidence: {out['confidence']*100:.2f}%")
                fig, ax = plt.subplots()
                ax.bar(range(len(out['probs'])), out['probs'])
                ax.set_xlabel("Class")
                ax.set_ylabel("Probability")
                st.pyplot(fig)
        except Exception as e:
            st.error(str(e))

# ---- Upload CSV ----
elif mode == "Upload CSV (batch)":
    up = st.file_uploader("Upload CSV (rows of features, last col optional label)", type="csv")
    if up is not None:
        df = pd.read_csv(up, header=None)
        st.write("Loaded:", df.shape)

        try:
            model, scaler = _load()
            expected = scaler.n_features_in_
        except:
            expected = df.shape[1] - 1  # fallback guess

        if df.shape[1] == expected + 1:
            X = df.iloc[:, :-1].astype(float).values
            y = df.iloc[:, -1].astype(int).values
        else:
            X = df.astype(float).values
            y = None

        results = []
        if use_api:
            for i, row in df.iterrows():
                features = row[:-1].tolist() if df.shape[1] == expected + 1 else row.tolist()
                try:
                    r = requests.post(api_url, json={"features": features}, timeout=10)
                    results.append(r.json())
                except Exception as e:
                    results.append({"error": str(e)})
            st.write(results[:10])
        else:
            Xs = scaler.transform(X)
            preds = model.predict(Xs)
            probs = model.predict_proba(Xs).max(axis=1)
            out = pd.DataFrame({"pred": preds, "confidence": probs})
            st.dataframe(out.head(30))
            csv = out.to_csv(index=False).encode()
            st.download_button("Download results", csv, "predictions.csv")

# ---- Use Sample Row ----
elif mode == "Use sample row":
    sr = sample_row()
    if sr is None:
        st.warning("No sample found in Datasets/mitbih_test.csv")
    else:
        st.write(f"Loaded sample row with {len(sr)} values.")
        if st.button("Predict sample"):
            out = predict_row(sr)
            st.success(f"Prediction: {out['label']} (class {out['class']})")
            st.info(f"Confidence: {out['confidence']*100:.2f}%")
            fig, ax = plt.subplots()
            ax.plot(sr)
            ax.set_title("ECG Beat (sample)")
            st.pyplot(fig)

# ---- Quick EDA ----
st.markdown("---")
st.header("Quick EDA (Dataset)")
csv_path = ROOT / "Datasets" / "mitbih_test.csv"
if csv_path.exists():
    df_all = pd.read_csv(csv_path, header=None)
    st.write("Dataset shape:", df_all.shape)
    st.write("Class distribution:")
    labels = df_all.iloc[:, -1].astype(int)
    st.bar_chart(labels.value_counts())
else:
    st.info("Put mitbih_test.csv into Datasets/ and refresh.")
