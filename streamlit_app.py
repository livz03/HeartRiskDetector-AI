import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model and scaler
# -----------------------------
model = joblib.load("src/model_1_rf.pkl")
scaler = joblib.load("src/model_1_scaler.pkl")

st.title("❤️ Heart Risk Detector AI")
st.subheader("Machine Learning powered health screening")

st.write("This app predicts heart disease risk based on clinical inputs like Age, Cholesterol, BP, and ECG results.")

# -----------------------------
# Collect user inputs
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
ecg = st.selectbox("ECG Result", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
heartrate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)

# -----------------------------
# Encode categorical
# -----------------------------
sex_val = 1 if sex == "Male" else 0
ecg_map = {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}
ecg_val = ecg_map[ecg]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Risk"):
    input_data = pd.DataFrame([[age, sex_val, cholesterol, bp, ecg_val, heartrate]],
                              columns=["Age", "Sex", "Cholesterol", "RestingBP", "ECG", "HeartRate"])
    
    # Scale inputs
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of risk
    
    # Show result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f" High Risk Detected (Probability: {probability:.2f})")
        st.warning(" Advice: Please visit a hospital for a full checkup immediately.")
    else:
        st.success(f" Low Risk (Probability: {probability:.2f})")
        st.info(" Advice: Maintain healthy habits, regular exercise, and routine checkups.")

    # -----------------------------
    # Chart Visualization
    # -----------------------------
    labels = ["Low Risk", "High Risk"]
    values = [1 - probability, probability]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#6bd66b", "#ff6b6b"])
    ax.axis("equal")
    st.pyplot(fig)
