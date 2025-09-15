import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model():
    test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "mitbih_test.csv"))

    # check if file exists
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f" Test dataset not found at  {test_data_path}")
    
    print(f"Loading test  dataset from {test_data_path}...")
    try:
        test_df = pd.read_csv(test_data_path,header=None)
        print(f"Dataset loaded successfully with shape:{test_df.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load test dataset:{e}")

    

    

    #model_path="../models/mitbih_rf_model.pkl",
    #report_path="../reports/mitbih_classification_report.txt"

    """
    Evaluate trained model on mitbih test dataset.
    """

    #Load dataset
    #print("Loading test dataset from {test_data_path}...")
    #data = pd.read_csv(test_data_path)

    #Features and labels
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Set this variable to "heart" or "mitbih" to select the dataset
    #dataset = "mitbih"  # or "heart"
     

    # Path to trained model 
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","models","mitbih_rf_model.pkl"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first using train.py.")
    #test_path = os.path.abspath(test_data_path)
    #test_path = os.path.abspath(model_path)

    #test_df = pd.read_csv(test_data_path)

    # Features and labels
    #X_test = test_df.iloc[:, :-1]
    #y_test = test_df.iloc[:, -1]

    print("Loading trained model...")

    # Load the model
    #model = joblib.load(os.path.join(os.path.dirname(__file__),"..", "models", "mitbih_rf_model.pkl"))
    model = joblib.load(model_path) 


    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Evaluating model...")
    print(clf_report)

    
    print("confusion_matrix:")
    print(cm)

    print("Accuracy:")
    print(f"Accuracy: {acc:.4f}")
 
    ## Save report
    #with open(report_path, "w") as f:
    #    f.write(f"Accuracy: {acc:.4f}\n\n")
    #    f.write("Classification Report:\n")
    #    f.write(clf_report + "\n\n")
    #    f.write("Confusion Matrix:\n")
    #    f.write(str(cm) )
    #    t
    #    print(f"Evaluation report saved to {report_path}")


  
if __name__ == "__main__":
    
    evaluate_model()
