import joblib
import os

PRODUCTION_MODEL_PATH = "customer_churn_gbm_model.pkl"

def save_model(model):
    joblib.dump(model, PRODUCTION_MODEL_PATH)
    print(f"Model saved to {PRODUCTION_MODEL_PATH}")
