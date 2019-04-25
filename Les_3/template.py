# Note: implement the below classes...

# IMPORTS:

from math import tanh, log, exp
from copy import deepcopy


# FUNCTIONS:

## Copy previous definitions of activation-, loss- and derivative-functions


# CLASSES

class Layer():

    def __init__(self, next=None):
        # "next" optionally contains a reference to the next neural layer
        self.next = next

    def __str__(self):
        # returns an informative description
        if self.next == None:
            return 'None'
        else:
            return '\\\n' + str(self.next)

    def __add__(self, other):
        # "other" contains a reference to the neural layer to be concatenated
        result = deepcopy(self)
        result.append(deepcopy(other))
        return result

    def append(self, next):
        # "next" contains a reference to the neural layer to be appended
        if self.next == None:
            # if this is the last layer, append the next layer to this layer
            self.next = next
        else:
            # if there is a next layer already, pass the next layer to that
            self.next.append(next)


class FullLayer(Layer):

    def __init__(self, inputs, outputs, act_func=identity_act_func, next=None):
        # "inputs" equals the number of inputs
        # "outputs" equals the number of outputs
        # "act_func" contains a reference to the activation function
        # "next" optionally contains a reference to the next neural layer
        super().__init__(next)
        pass

    def __str__(self):
        # returns an informative description
        text = 'FullLayer():'
        text += '\n  - biases = ' + str(self.biases)
        text += '\n  - weights = ' + str(self.weights)
        text += '\n  - act_func = ' + self.act_func.__name__
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

    def fit(self, xs, ys, alpha=0, epochs=100):
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # "epochs" equals the number of epochs
        pass


class LossLayer(Layer):

    def __init__(self, loss_func=quadratic_loss_func):
        # "loss_func" contains a reference to the loss-function
        super().__init__()
        pass

    def __str__(self):
        # returns an informative description
        text = 'LossLayer():'
        text += '\n  - loss_func = ' + self.loss_func.__name__
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
