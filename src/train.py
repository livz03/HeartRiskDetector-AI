import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocessor import build_preprocessor, split_Xy
from .config import MODEL_PATH

def train_pipeline(df):
    preprocessor = build_preprocessor(df)
    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = split_Xy(df)
    pipe.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe, (X_test, y_test)
