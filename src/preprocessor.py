import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, target_column: str):
    """Preprocess dataset : split into train/test and scale features.
    Args:
        df (pd.DataFrame): Input dataset.
        target_column (str): Name of the target column.
    Returns:
         X_train, X_test, y_train, y_test, scaler
    """
    try:
        # Handle missing values
        df = df.dropna()  # or use imputer to fill missing values

        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print("Data preprocessing completed successfully.")
        return X_train, X_test, y_train, y_test, scaler
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None, None, None, None, None
