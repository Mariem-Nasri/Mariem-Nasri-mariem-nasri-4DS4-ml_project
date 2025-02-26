# tests/test_preprocessing.py
import pytest
from src.preprocessing import preprocess_data  # Replace with actual function

def test_preprocess_data():
    input_data = [1, 2, 3]
    expected_output = [0.0, 0.5, 1.0]
    output = preprocess_data(input_data)
    assert output == expected_output
