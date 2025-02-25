# src/config.py
import os

# Define directories for processed data and models
PROCESSED_DATA_DIR = "processed_data"
MODEL_DIR = "models"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Define paths for saving data
DATA_PATHS = {
    'X_train': os.path.join(PROCESSED_DATA_DIR, "X_train.pkl"),
    'X_test': os.path.join(PROCESSED_DATA_DIR, "X_test.pkl"),
    'y_train': os.path.join(PROCESSED_DATA_DIR, "y_train.pkl"),
    'y_test': os.path.join(PROCESSED_DATA_DIR, "y_test.pkl"),
    'model': os.path.join(MODEL_DIR, "trained_model.pkl")
}
