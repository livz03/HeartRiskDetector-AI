from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np
from pathlib import Path

# Load model + scaler
ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "models" / "mitbih_1_rf.pkl")
scaler = joblib.load(ROOT / "models" / "mitbih_1_scaler.pkl")

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get("features", None)

        if features is None:
            return jsonify({"error": "Missing 'features' list"}), 400

        # Ensure numpy float array, reshape, scale
        features = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)

        pred = int(model.predict(features_scaled)[0])
        probs = model.predict_proba(features_scaled)[0].tolist()

        out = {
            "class": pred,
            "label": f"Class {pred}",
            "confidence": max(probs),
            "probs": probs
        }
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
