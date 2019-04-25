# Note: implement the below classes...

# IMPORTS:

from math import tanh, exp


# FUNCTIONS:

# +----------------------------+
# | Some activation-functions: |
# +----------------------------+

def identity_act_func(a):
    return None

def signum_act_func(a):
    return None

def tanh_act_func(a):
    return None

def softsign_act_func(a):
    return None

# +----------------------+
# | Some loss-functions: |
# +----------------------+

def quadratic_loss_func(y_hat, y):
    return None

def absolute_loss_func(y_hat, y):
    return None

def perceptron_loss_func(y_hat, y):
    return None

# +-------------------------------+
# | The derivative of a function: |
# +-------------------------------+

def derivative(function, delta_x=0):
    def gradient(x):
        pass
        return None
    return gradient


# CLASSES

class Neuron():

    def __init__(self, dim=2, act_func=identity_act_func, loss_func=quadratic_loss_func):
        # "dim" equals the dimensionality of the attributes
        # "act_func" contains a reference to the activation function
        # "loss_func" contains a reference to the loss function
        pass

    def __str__(self):
        # Returns an informative description
        result = 'Neuron():'
        result += '\n  - bias = ' + str(self.bias)
        result += '\n  - weights = ' + str(self.weights)
        result += '\n  - act_func = ' + str(self.act_func)
        result += '\n  - loss_func = ' + str(self.loss_func)
        return result

    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        pass
        return None

    def loss(self, x, y):
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct outcome
        pass
        return None

    def train(self, x, y, alpha=0):
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct outcome
        # "alpha" is the learning rate; choose a suitable default value
        pass

    def fit(self, xs, ys, alpha=0, epochs=100):
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # "epochs" equals the number of epochs
        pass
