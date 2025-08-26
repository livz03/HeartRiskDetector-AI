# preprocessor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath: str):
    """
    Load dataset, preprocess it, and return X_train, X_test, y_train, y_test
    Target column is set to 'heartdisease'
    """

    # Load dataset
    data = pd.read_csv(filepath)

    # Make sure target column exists
    if 'heartdisease' not in data.columns:
        raise ValueError("❌ Target column 'heartdisease' not found in dataset. "
                         "Check your CSV headers.")

    # Features and target
    X = data.drop('heartdisease', axis=1)
    y = data['heartdisease']

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
