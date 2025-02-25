from sklearn.ensemble import GradientBoostingClassifier
import joblib
from src.config import DATA_PATHS  # Assuming paths are imported from main.py


def train_model(X_train_scaled_smote, y_train_smote):
    # Define the model
    best_params = {
        "max_depth": 10,
        "max_features": None,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 5,
    }

    gbm = GradientBoostingClassifier(**best_params)
    gbm.fit(X_train_scaled_smote, y_train_smote)

    # Save the model
    joblib.dump(gbm, DATA_PATHS["model"])
    return gbm
