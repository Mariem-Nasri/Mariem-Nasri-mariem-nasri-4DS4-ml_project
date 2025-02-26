import pytest
import joblib
import sys
import os
from unittest.mock import Mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_data

# Import modules
from src.prepare import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.save import save_model
from src.load import load_model
from src.predict import make_prediction
from src.config import DATA_PATHS

# Suppress specific warnings
pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

def test_prepare_data():
    """Test the prepare_data function."""
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Assertions to ensure data is loaded correctly
    assert X_train is not None and X_test is not None
    assert y_train is not None and y_test is not None
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0
    
    # Save the prepared data for use in other tests
    joblib.dump(X_train, DATA_PATHS["X_train"])
    joblib.dump(X_test, DATA_PATHS["X_test"])
    joblib.dump(y_train, DATA_PATHS["y_train"])
    joblib.dump(y_test, DATA_PATHS["y_test"])

def test_train_model():
    """Test the train_model function."""
    X_train = joblib.load(DATA_PATHS["X_train"])
    y_train = joblib.load(DATA_PATHS["y_train"])
    
    # Train the model
    model = train_model(X_train, y_train)
    assert model is not None
    
    # Save the trained model for use in other tests
    joblib.dump(model, DATA_PATHS["model"])

def test_evaluate_model():
    """Test the evaluate_model function."""
    model = joblib.load(DATA_PATHS["model"])
    X_test = joblib.load(DATA_PATHS["X_test"])
    y_test = joblib.load(DATA_PATHS["y_test"])
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Assertions for required metrics
    assert "accuracy" in metrics and metrics["accuracy"] > 0
    assert "precision" in metrics and metrics["precision"] > 0
    assert "recall" in metrics and metrics["recall"] > 0
    assert "f1_score" in metrics and metrics["f1_score"] > 0

def test_save_and_load_model():
    """Test the save_model and load_model functions."""
    model = joblib.load(DATA_PATHS["model"])
    
    # Save the model
    save_model(model)
    
    # Load the model
    loaded_model = load_model()
    assert loaded_model is not None

def test_make_prediction():
    """Test the make_prediction function."""
    model = joblib.load(DATA_PATHS["model"])
    assert model is not None
    
    # Create a mock logger
    mock_logger = Mock()
    
    # Call the make_prediction function with the mock logger
    make_prediction(model, logger=mock_logger)
    
    # Assert that the logger was called
    mock_logger.info.assert_called()