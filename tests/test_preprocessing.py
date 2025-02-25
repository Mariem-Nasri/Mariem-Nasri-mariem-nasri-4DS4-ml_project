# tests/test_preprocessing.py
import pytest
from src.preprocessing import preprocess_data  # Replace with actual function

def test_preprocess_data():
    # Test with sample data
    input_data = [1, 2, 3]
    expected_output = [0.0, 0.5, 1.0]  # Example output after scaling
    assert preprocess_data(input_data) == expected_output
