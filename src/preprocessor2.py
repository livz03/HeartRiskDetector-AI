# src/preprocessor.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def find_dataset(filename="mitbih_test.csv"):
    root = Path(__file__).resolve().parent.parent
    p1 = root / "Datasets" / filename
    p2 = root / "datasets" / filename
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"{filename} not found in Datasets/ or datasets/")

def load_mitbih(filename="mitbih_test.csv", header=None):
    path = find_dataset(filename)
    df = pd.read_csv(path, header=header)
    return df

def split_features_labels(df):
    # last column is assumed label
    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].astype(int).values
    return X, y

def train_test_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler
