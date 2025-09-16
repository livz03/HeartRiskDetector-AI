# src/streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict2 import predict_row

st.set_page_config(page_title="Heart Risk Detector", layout="wide")
st.title("Heart / ECG Risk Detector")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Input mode", ["Paste single row", "Upload CSV (batch)"])

# Single row prediction
if mode == "Paste single row":
    txt = st.text_area("Paste 187 comma-separated features (no label):")
    if st.button("Predict"):
        try:
            features = [float(x.strip()) for x in txt.split(",") if x.strip() != ""]
            out = predict_row(features)
            st.success(f"Prediction: {out['label']} (class {out['class']})")
            st.info(f"Advice: {out['advice']}")
            st.info(f"Confidence: {out['confidence']*100:.2f}%")
            # Bar chart for probabilities
            fig, ax = plt.subplots()
            ax.bar(range(len(out['probs'])), out['probs'])
            ax.set_xlabel("Class")
            ax.set_ylabel("Probability")
            st.pyplot(fig)
        except Exception as e:
            st.error(e)

# Batch CSV prediction
elif mode == "Upload CSV (batch)":
    up = st.file_uploader("Upload CSV (rows of features)", type="csv")
    if up:
        df = pd.read_csv(up, header=None)
        st.write("Loaded dataset:", df.shape)
        results = []
        for i, row in df.iterrows():
            try:
                r = predict_row(row.tolist())
                results.append(r)
            except:
                continue
        out_df = pd.DataFrame(results)
        st.dataframe(out_df)
        csv = out_df.to_csv(index=False).encode()
        st.download_button("Download Predictions CSV", csv, "predictions.csv")

# Quick dataset visualization
st.markdown("---")
st.header("Quick EDA")
try:
    df_all = pd.read_csv("../Datasets/mitbih_test.csv", header=None)
    st.write("Dataset shape:", df_all.shape)
    labels = df_all.iloc[:, -1].astype(int)
    st.bar_chart(labels.value_counts())
except:
    st.info("Put mitbih_test.csv in Datasets/ folder.")
