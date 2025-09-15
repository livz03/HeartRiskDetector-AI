# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------- Preprocessor Function ---------------- #
def load_and_preprocess_data(data):
    """
    Load dataset, preprocess it, and return X_train, X_test, y_train, y_test
    Target column must be 'heartdisease'
    """
    # Handle file uploaded via Streamlit or filepath string
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = pd.read_csv(data)

    if 'heartdisease' not in df.columns:
        raise ValueError("❌ Target column 'heartdisease' not found in dataset.")

    X = df.drop('heartdisease', axis=1)
    y = df['heartdisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns  # Return columns for dynamic input

# ---------------- Heart Risk Detector Page ---------------- #
def heart_risk_detector():
    st.title("💓 Heart Risk Detector AI")
    st.write("Predict **heart disease risk** based on health data.")

    uploaded_file = st.file_uploader("..Datasets\\heart.csv", type=["csv"])

    if uploaded_file is not None:
        try:
            # Preprocess data
            X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess_data(uploaded_file)

            # Train Logistic Regression Model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.subheader("📊 Model Performance")
            st.write(f"✅ Accuracy: {acc:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # User Input for Prediction
            st.subheader("🧑‍⚕️ Predict Your Risk")
            st.write("Enter your health details:")
            user_input = []
            for col in feature_cols:
                val = st.number_input(f"{col}", value=0)
                user_input.append(val)

            if st.button("🔍 Predict"):
                prediction = model.predict([user_input])[0]
                st.success("💖 No Heart Disease Risk!" if prediction == 0 else "⚠️ High Risk of Heart Disease!")

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- Main App ---------------- #
def main():
    st.set_page_config(page_title="Heart Risk Detector AI", layout="centered")
    heart_risk_detector()

if __name__ == "__main__":
    main()
