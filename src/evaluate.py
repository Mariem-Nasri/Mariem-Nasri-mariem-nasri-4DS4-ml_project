# evaluate.py
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from elasticsearch import Elasticsearch


es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "nvvDaamm0aYxHehoAHsj"),  # Use `basic_auth` instead of `http_auth`
    verify_certs=False
)


class ElasticsearchHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        es.index(index='mlflow-metrics', document={'message': log_entry})
        if not es.indices.exists(index='mlflow-metrics'):
            es.indices.create(index='mlflow-metrics')


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
    es_handler = ElasticsearchHandler()
    logger.addHandler(es_handler)
    logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")  # ADD F1-score ✅

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}  # RETURN F1 ✅
