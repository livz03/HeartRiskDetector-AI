from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('../models/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    vals = [float(x) for x in request.form.values()]
    prediction = model.predict([vals])
    return render_template('index.html', prediction=f'Risk: {"Yes" if prediction[0]==1 else "No"}')

if __name__ == "__main__":
    app.run(debug=True)