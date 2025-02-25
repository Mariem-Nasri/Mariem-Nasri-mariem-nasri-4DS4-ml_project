# src/main.py
import argparse
import joblib
from src.config import DATA_PATHS  # Import DATA_PATHS from config.py
from src.prepare import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.save import save_model
from src.load import load_model

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--save", action="store_true", help="Save the trained model")
    parser.add_argument("--load", action="store_true", help="Load a saved model")
    args = parser.parse_args()

    gbm = None
    X_train_scaled_smote_df = X_test_scaled_smote_df = y_train_smote_df = y_test_smote_df = None

    # Step 1: Prepare data if needed
    if args.prepare:
        print("Preparing data...")
        X_train_scaled_smote_df, X_test_scaled_smote_df, y_train_smote_df, y_test_smote_df = prepare_data()
        joblib.dump(X_train_scaled_smote_df, DATA_PATHS['X_train'])
        joblib.dump(X_test_scaled_smote_df, DATA_PATHS['X_test'])
        joblib.dump(y_train_smote_df, DATA_PATHS['y_train'])
        joblib.dump(y_test_smote_df, DATA_PATHS['y_test'])
        print("Data preparation complete.")

    # Step 2: Train model if needed
    if args.train:
        print("Training model...")
        try:
            X_train_scaled_smote_df = joblib.load(DATA_PATHS['X_train'])
            y_train_smote_df = joblib.load(DATA_PATHS['y_train'])
        except FileNotFoundError:
            print("Error: Data files not found. Run --prepare first.")
            return
        gbm = train_model(X_train_scaled_smote_df, y_train_smote_df)
        joblib.dump(gbm, DATA_PATHS['model'])
        print("Model training complete.")

    # Step 3: Evaluate model if needed
    if args.evaluate:
        print("Evaluating model...")
        try:
            gbm = joblib.load(DATA_PATHS['model'])
            X_test_scaled_smote_df = joblib.load(DATA_PATHS['X_test'])
            y_test_smote_df = joblib.load(DATA_PATHS['y_test'])
        except FileNotFoundError:
            print("Error: Model or data files not found. Run --train or --prepare first.")
            return
        metrics = evaluate_model(gbm, X_test_scaled_smote_df, y_test_smote_df)
        print("Model evaluation complete.")
        print("Metrics:", metrics)

    # Step 4: Save model if needed
    if args.save:
        print("Saving model...")
        try:
            gbm = joblib.load(DATA_PATHS['model'])
        except FileNotFoundError:
            print("Error: No model found. Run --train first.")
            return
        save_model(gbm)
        print("Model saved.")

    # Step 5: Load model if needed
    if args.load:
        print("Loading saved model...")
        gbm = load_model()
        if gbm is None:
            print("Error: No saved model found. Train and save a model first.")
            return
        print("Model loaded successfully.")

if __name__ == "__main__":
    main()