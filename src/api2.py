# src/api2.py
from flask import Flask, request, jsonify
from pathlib import Path
import sys

# add src to path to import predict2
sys.path.append(str(Path(__file__).resolve().parent))

from predict2 import predict_row

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Expect JSON: { "features": [f1, f2, ..., f187] }
    Returns { "class": int, "label": str, "confidence": float, "probs": [...] }
    """
    data = request.get_json(force=True)
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features' (list of numbers)"}), 400

    features = data["features"]
    try:
        out = predict_row(features)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # for local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
