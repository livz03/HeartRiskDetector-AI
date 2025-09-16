# src/predict2.py
import joblib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def _load():
    model = joblib.load(ROOT.parent / "models" / "model_1_rf.pkl")
    scaler = joblib.load(ROOT.parent / "models" / "mode_1_scaler.pkl")
    return model, scaler

def predict_row(row):
    """
    row: list of features (length 187)
    returns: dict with class, label, confidence, probs
    """
    model, scaler = _load()
    X = scaler.transform(np.array(row).reshape(1, -1))
    pred_class = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = probs.max()
    
    # Mapping to labels and advice
    label_map = {0: "Safe", 1: "Medium Risk", 2: "High Risk"}
    advice_map = {0: "Stay safe", 1: "Monitor health", 2: "Visit hospital"}
    
    return {
        "class": int(pred_class),
        "label": label_map[int(pred_class)],
        "advice": advice_map[int(pred_class)],
        "confidence": float(confidence),
        "probs": probs.tolist()
    }
