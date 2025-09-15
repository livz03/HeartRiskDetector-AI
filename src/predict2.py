# src/predict.py
import os 
from pathlib import Path
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mitbih_1_rf.pkl"
SCALER_PATH = ROOT / "models" / "mitbih_1_scaler.pkl"

def _load():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found. Run src/train.py first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_row(features):
    """
    features: list/array length 187
    returns: dict { 'class': int, 'label': str, 'probs': [..], 'confidence': float }
    """
    model, scaler = _load()
    arr = np.array(features, dtype=float).reshape(1, -1)
    if hasattr(scaler, "n_features_in_"):
        expected = scaler.n_features_in_
        if arr.shape[1] != expected:
            raise ValueError(f"Expected {expected} features, got {arr.shape[1]}")
    arr_s = scaler.transform(arr)
    probs = model.predict_proba(arr_s)[0]
    cls = int(model.predict(arr_s)[0])

    # mapping: class 0 => Low/Normal, others => Abnormal/High (adjust as needed)
    label = "Low/Normal" if cls == 0 else "Abnormal/High"
    return {
        "class": cls,
        "label": label,
        "probs": probs.tolist(),
        "confidence": float(max(probs))
    }

