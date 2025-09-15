# src/train.py
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessor2 import load_mitbih, split_features_labels, train_test_scale

def main():
    print("Loading dataset...")
    df = load_mitbih(header=None)
    X, y = split_features_labels(df)
    print("Shapes:", X.shape, y.shape)

    print("Train/test split and scaling...")
    X_train, X_test, y_train, y_test, scaler = train_test_scale(X, y, test_size=0.2)

    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Validating on hold-out set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "mitbih_1_rf.pkl")
    joblib.dump(scaler, out_dir / "mitbih_1-scaler.pkl")
    print(f"Saved model and scaler to {out_dir}")

if __name__ == "__main__":
    main()
