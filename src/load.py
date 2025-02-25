import joblib
import os

PRODUCTION_MODEL_PATH = "customer_churn_gbm_model.pkl"

def load_model():
    if os.path.exists(PRODUCTION_MODEL_PATH):
        return joblib.load(PRODUCTION_MODEL_PATH)
    else:
        print(f"No model found at {PRODUCTION_MODEL_PATH}")
        return None
