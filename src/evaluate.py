# evaluate.py
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    # Log metrics to Elasticsearch
    logger = logging.getLogger(__name__)
    logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")  # ADD F1-score ✅

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}  # RETURN F1 ✅
