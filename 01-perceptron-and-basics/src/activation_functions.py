import numpy as np

def step(x):
    """
    Step activation: The simplest decision function.
    
    Real-world: A light switch. On if positive, off if negative.
    Used in perceptrons for final classification.
    
    Returns 1 if x > 0, else 0.
    """
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    """
    Sigmoid activation: Smooth "S-shaped" curve.
    
    Real-world: A dimmer switch. Gives smooth transition from 0 to 1.
    Also interpretable as probability (output between 0 and 1).
    
    Mathematical formula: 1 / (1 + e^(-x))
    
    Why it's useful:
    - Smooth (continuous) → we can calculate derivatives
    - Bounded between 0 and 1 → good for probabilities
    - Used in output layers for binary classification
    
    Parameters:
    - x: Input (can be a number or array)
    
    Returns:
    - Sigmoid of x, between 0 and 1
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a):
    """
    Derivative of the sigmoid function.
    
    Why we need this: For backpropagation in multi-layer networks.
    Tells us how much a small change in input affects the output.
    
    Real-world: The slope of the dimmer switch. 
    When the light is at half brightness (a=0.5), the slope is steepest (0.25).
    When the light is fully on or off, the slope is near zero.
    
    Mathematical formula: a * (1 - a)
    (where a = sigmoid(x))
    
    Parameters:
    - a: Output of sigmoid function (already computed)
    
    Returns:
    - Derivative value at that point
    """
    return a * (1 - a)