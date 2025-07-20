from typing import *

def initialize_weights(n_in, n_hidden, n_out):
    """
    Initialize weights and biases for a simple neural network with one hidden layer.
    
    Parameters:
    - n_in: Number of neurons in the input layer.
    - n_hidden: Number of neurons in the hidden layer.
    - n_out: Number of neurons in the output layer.
    
    Returns:
    - weights: A dictionary containing the weights of the network.
               Keys: 'input_hidden', 'hidden_output'.
               Values: Numpy arrays representing the weights.
    - biases: A dictionary containing the biases of the network.
              Keys: 'hidden', 'output'.
              Values: Numpy arrays representing the biases.
    """
    weights = {
        'input_hidden': np.random.normal(0, 0.01, (n_hidden, n_in)),
        'hidden_output': np.random.normal(0, 0.01, (n_out, n_hidden))
    }
    biases = {
        'hidden': np.zeros((n_hidden, 1)),
        'output': np.zeros((n_out, 1))
    }
    
    return weights, biases

### Unit tests below ###
def check(candidate):
    assert isinstance(candidate(3, 4, 2)[0]['input_hidden'], np.ndarray)
    assert candidate(3, 4, 2)[0]['input_hidden'].shape == (4, 3)
    assert isinstance(candidate(3, 4, 2)[0]['hidden_output'], np.ndarray)
    assert candidate(3, 4, 2)[0]['hidden_output'].shape == (2, 4)
    assert isinstance(candidate(3, 4, 2)[1]['hidden'], np.ndarray)
    assert candidate(3, 4, 2)[1]['hidden'].shape == (4, 1)
    assert isinstance(candidate(3, 4, 2)[1]['output'], np.ndarray)
    assert candidate(3, 4, 2)[1]['output'].shape == (2, 1)
    assert np.all(candidate(3, 4, 2)[1]['hidden'] == 0)
    assert np.all(candidate(3, 4, 2)[1]['output'] == 0)

def test_check():
    check(initialize_weights)
