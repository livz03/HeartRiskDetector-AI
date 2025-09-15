# main.py
import streamlit as st
from preprocessor import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    st.set_page_config(page_title="Heart Risk Detector AI", layout="centered")
    st.title("💓 Heart Risk Detector AI")
    st.write("Predict **heart disease risk** based on health data.")

    uploaded_file = st.file_uploader("📂 Upload your Heart Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            # Preprocess data
            X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess_data(uploaded_file)

            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.subheader("📊 Model Performance")
            st.write(f"✅ Accuracy: {acc:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # User input for prediction
            st.subheader("🧑‍⚕️ Predict Your Risk")
            user_input = []
            for col in feature_cols:
                val = st.number_input(f"{col}", value=0)
                user_input.append(val)

            if st.button("🔍 Predict"):
                prediction = model.predict([user_input])[0]
                st.success("💖 No Heart Disease Risk!" if prediction == 0 else "⚠️ High Risk of Heart Disease!")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
