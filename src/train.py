from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os
from src.config import DATA_PATHS  # Assuming paths are imported from main.py


def train_model(X_train_scaled_smote, y_train_smote):
    # Define the model parameters
    best_params = {
        "max_depth": 10,
        "max_features": None,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 5,
    }

    # Initialize the model
    gbm = GradientBoostingClassifier(**best_params)
    try:
        # Fit the model to the training data
        gbm.fit(X_train_scaled_smote, y_train_smote)
    except ValueError as e:
        print(f"Error in training the model: {e}")
        return None
    # Ensure the directory exists before saving the model
    model_path = DATA_PATHS.get("model", "")
    if model_path:
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    else:
        print("Error: Model path is not defined.")
        return None

    # Save the trained model
    try:
        joblib.dump(gbm, model_path)
    except Exception as e:
        print(f"Error saving the model: {e}")
        return None

    return gbm
