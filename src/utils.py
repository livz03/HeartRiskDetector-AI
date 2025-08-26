# src/utils.py

def format_input(user_input_dict):
    """
    Converts user input dictionary into list for prediction
    """
    features_order = ['age', 'cholesterol', 'blood_pressure', 'glucose', 'bmi']  # adjust based on your CSV
    input_list = [float(user_input_dict.get(feat, 0)) for feat in features_order]
    return input_list
