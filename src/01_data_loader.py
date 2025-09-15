import pandas as pd

def load_data(filepath: str):
    """_summary_
    Load dataset from a CSV file and return a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None