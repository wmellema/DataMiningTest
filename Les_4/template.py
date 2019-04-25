# Note: implement the below classes...

# IMPORTS:

from math import tanh, log, exp
from random import uniform
from copy import deepcopy


# FUNCTIONS:

## Insert previous definitions of activation-, loss- and derivative-functions!

def logistic_act_func(a):
    return None

def softplus_act_func(a):
    return None

def relu_act_func(a):
    return None

def crossentropy_loss_func(y_hat, y):
    return None


# CLASSES

## Insert previous definitions of the Layer-, FullLayer- and LossLayer-classes!

class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        # returns an informative description
        text = 'SoftmaxLayer()'
        text += '\n  - next = ' + super().__str__()
        return text

    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        # returns the predicted outcome
        pass
        return None

    def loss(self, x, y):
        # "x" contains a list with the attributes of a single instance
        # "y" contains a list with the corresponding correct outcomes
        # returns the loss of the prediction
        pass
        return None

    def train(self, x, y, alpha=0):
        # "x" contains a list with the attributes of a single instance
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # returns the gradient of the loss with respect to the input x
        pass
        return None
