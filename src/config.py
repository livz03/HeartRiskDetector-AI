import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "datasets", "heart.csv")   # adjust if needed
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")

# IMPORTANT: set to your label column present in datasets/heart.csv
LABEL_COL = "HeartDisease"   # change to "target" if your file uses that
TEST_SIZE = 0.2
RANDOM_STATE = 42
