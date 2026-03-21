import sys
import os
import numpy as np
import pytest

# Add the parent directory (01-perceptron-and-basics) to the path
# This allows importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#now we can import from src
from src.perceptron import Perceptron

def test_and():
    """Test that the perceptron learns the AND function."""
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.float64)

    p = Perceptron(learning_rate=0.1, n_iterations=10)
    p.fit(X, y)
    preds = p.predict(X)
    
    #assert all predictions match the true labels
    assert np.array_equal(preds, y), f"AND test failed: predicted {preds}, expected {y}"

def test_or():
    """Test that the perceptron learns the OR function."""
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
    y = np.array([0, 1, 1, 1], dtype=np.float64)

    p = Perceptron(learning_rate=0.1, n_iterations=10)
    p.fit(X, y)
    preds = p.predict(X)
    
    assert np.array_equal(preds, y), f"OR test failed: predicted {preds}, expected {y}"

if __name__ == "__main__":
    pytest.main([__file__])