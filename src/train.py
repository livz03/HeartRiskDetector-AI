import os
import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

##File paths
#DATA_Path = 'data/mitbih_test.csv'
#MODEL_Path = 'models/mitbih_rf_model.pkl'
#SCALER_Path = 'models/scaler.pkl'


def train_model():
        base_dir = os.path.dir(__file__)
        train_data_path = os.path.join(base_dir,"..","Datasets","mitbih_test.csv")
def load_data(filepath: str):
              """Load dataset of mitbih """
              df = pd.read_csv('../Datasets/mitbih_test.csv')
              X = df.iloc[:, :-1]
              y = df.iloc[:, -1]
              return X, y


def train_model(X_train, y_train):
        """Train a RadomForestClassifier model"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model



 
def main():
        DATA_Path = '../Datasets/mitbih_test.csv'
        print("Loading data...")
        X, y = load_data(DATA_Path)

        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        print("Preprocessing data...")
        X_train_scaled, scaler = preprocess_data(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training model...")
        model = train_model(X_train_scaled, y_train)

        os.makedirs("../models", exist_ok=True)
        joblib.dump(model,"models/mitbih_rf_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print(f"Model saved to {MODEL_Path}")
        print(f"Scaler saved to {SCALER_Path}")

        if __name__ == "__main__":
            main()
