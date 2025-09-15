`lambda import joblib
import pandas as pd

def load_pipeline(model_path: str):
    return joblib.load(model_path)

def predict_single(pipe, sample_dict: dict):
    X_new = pd.DataFrame([sample_dict])
    return int(pipe.predict(X_new)[0]), pipe.predict_proba(X_new)[0].tolist()
