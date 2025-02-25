import mlflow
import mlflow.sklearn
import argparse
import joblib
import logging
import numpy as np  # Add missing import
from src.config import DATA_PATHS
from src.prepare import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.save import save_model
from src.load import load_model


def main():
    # Ensure previous run is closed
    if mlflow.active_run():
        mlflow.end_run()

    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Customer_Churn_Experiment")

    # Set up logging
    logging.basicConfig(filename='main.log', level=logging.DEBUG)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--save", action="store_true", help="Save the trained model")
    parser.add_argument("--load", action="store_true", help="Load a saved model")
    parser.add_argument("--predict", action="store_true", help="Make predictions")  # Add this line
    args = parser.parse_args()

    gbm = None
    X_train_scaled_smote_df = X_test_scaled_smote_df = y_train_smote_df = (
        y_test_smote_df
    ) = None

    # Step 1: Prepare data if needed
    if args.prepare:
        print("Preparing data...")
        (
            X_train_scaled_smote_df,
            X_test_scaled_smote_df,
            y_train_smote_df,
            y_test_smote_df,
        ) = prepare_data()
        joblib.dump(X_train_scaled_smote_df, DATA_PATHS["X_train"])
        joblib.dump(X_test_scaled_smote_df, DATA_PATHS["X_test"])
        joblib.dump(y_train_smote_df, DATA_PATHS["y_train"])
        joblib.dump(y_test_smote_df, DATA_PATHS["y_test"])
        print("Data preparation complete.")

    # Step 2: Train model if needed
    if args.train:
        print("Training model...")
        try:
            X_train_scaled_smote_df = joblib.load(DATA_PATHS["X_train"])
            y_train_smote_df = joblib.load(DATA_PATHS["y_train"])
        except FileNotFoundError:
            print("Error: Data files not found. Run --prepare first.")
            return

        # Start MLflow run for training
        with mlflow.start_run() as run:
            gbm = train_model(X_train_scaled_smote_df, y_train_smote_df)
            joblib.dump(gbm, DATA_PATHS["model"])

            # Log model and parameters to MLflow
            mlflow.log_param("model_type", "Gradient Boosting")
            mlflow.log_param("data_version", "v1")

            # Log model
            mlflow.sklearn.log_model(gbm, "model")

            print("Model training complete.")

    # Step 3: Evaluate model if needed
    if args.evaluate:
        print("Evaluating model...")
        try:
            gbm = joblib.load(DATA_PATHS["model"])
            X_test_scaled_smote_df = joblib.load(DATA_PATHS["X_test"])
            y_test_smote_df = joblib.load(DATA_PATHS["y_test"])
        except FileNotFoundError:
            print("Error: Model or data files not found. Run --train or --prepare first.")
            return

        metrics = evaluate_model(gbm, X_test_scaled_smote_df, y_test_smote_df)
        print("Evaluation Metrics:", metrics)  # Debugging

        # Log evaluation metrics
        with mlflow.start_run():
            mlflow.log_metric("accuracy", metrics.get("accuracy", 0))
            mlflow.log_metric("precision", metrics.get("precision", 0))
            mlflow.log_metric("recall", metrics.get("recall", 0))

        print("Model evaluation complete.")
        print("Metrics:", metrics)

    # Step 4: Save model if needed
    if args.save:
        print("Saving model...")
        try:
            gbm = joblib.load(DATA_PATHS["model"])
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

    # Step 6: Make predictions if needed
    if args.predict:
        print("Making predictions...")
        try:
            gbm = joblib.load(DATA_PATHS["model"])
        except FileNotFoundError:
            print("Error: No model found. Run --train first.")
            return

        sample_data = np.array([[100, 1, 25, 150, 45.5, 130, 35.7, 120, 30.2, 30, 10.5, 2, 1, 30.0]])
        prediction = gbm.predict(sample_data)
        probability = gbm.predict_proba(sample_data)[0][1]
        print(f"Prediction: {prediction[0]}, Churn Probability: {probability:.4f}")


if __name__ == "__main__":
    main()
