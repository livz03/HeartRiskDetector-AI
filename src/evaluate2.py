# src/evaluate.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def find_artifacts():
    root = Path(__file__).resolve().parent.parent
    model_p = root / "models" / "mitbih_1_rf.pkl"
    scaler_p = root / "models" / "mitbih_1_scaler.pkl"
    return model_p if model_p.exists() else None, scaler_p if scaler_p.exists() else None

def main():
    model_p, scaler_p = find_artifacts()
    if model_p is None or scaler_p is None:
        raise FileNotFoundError("Model/scaler not found. Run src/train.py first.")

    df_path = Path(__file__).resolve().parent.parent / "Datasets" / "mitbih_test.csv"
    df = pd.read_csv(df_path, header=None)
    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].astype(int).values

    model = joblib.load(model_p)
    scaler = joblib.load(scaler_p)

    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Classification report:")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    main()
    
