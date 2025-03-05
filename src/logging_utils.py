# src/logging_utils.py
import mlflow
from elasticsearch import Elasticsearch
from mlflow.tracking import MlflowClient

# Set up Elasticsearch connection

es = Elasticsearch(
    "https://localhost:9200",
    http_auth=("elastic", "elastic"),
    verify_certs=False  # Only if you are using self-signed certificates
)
if es.ping():
    print("Connected to Elasticsearch!")
else:
    print("Could not connect to Elasticsearch.")
# Define the index name in Elasticsearch where logs will be stored
index_name = "mlflow-logs"

# Function to log metrics, parameters, and artifacts to Elasticsearch
def log_to_elasticsearch(run_id, metrics, params, tags, artifacts):
    log_data = {
        "run_id": run_id,
        "metrics": metrics,
        "params": params,
        "tags": tags,
        "artifacts": artifacts
    }
    
    # Send log data to Elasticsearch
    es.index(index=index_name, document=log_data)
    print(f"Logged data for run {run_id} to Elasticsearch")
