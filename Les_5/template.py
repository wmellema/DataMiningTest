# Note: implement the below classes...

# IMPORTS:

from math import sqrt, tanh, log, exp
from random import uniform, sample
from copy import deepcopy


# FUNCTIONS:

## Insert previous definitions of activation-, loss- and derivative-functions!


# CLASSES

class Layer():

    def __init__(self, next=None):
        # "next" optionally contains a reference to the next neural layer
        self.next = next

    def __str__(self):
        # returns an informative description
        if self.next is None:
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
        if self.next is None:
            # if this is the last layer, append the next layer to this layer
            self.next = next
        else:
            # if there is a next layer already, pass the next layer to that
            self.next.append(next)


class InputLayer(Layer):
    
    def __init__(self):
        super().__init__()

    def __str__(self):
        # returns an informative description
        text = 'InputLayer()'
        text += '\n  - next = ' + super().__str__()
        return text
    
    def predict(self, xs):
        # Wrapper function to return prediction
        predictions, _, _ = self.next.fit(xs)
        return predictions

    def loss(self, xs, ys):
        # Wrapper function to return losses
        _, losses, _ = self.next.fit(xs, ys)
        return losses

    def gradient(self, xs, ys):
        # Wrapper function to return gradients
        _, _, gradients = self.next.fit(xs, ys, 0.0)
        return gradients

    def fit(self, xs, ys, alpha=0.001, epochs=100, size=1):
        # "xs" contains a nested list with the attributes of all instances
        # "ys" contains a nested list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # "epochs" equals the number of epochs (default 100)
        # "size" equals the number of instances in a mini-batch (default 1)
        for epoch in range(epochs):
            instances = set(range(len(xs)))
            while len(instances) >= size:
                batch = sample(instances, size)
                instances = instances.difference(batch)
                self.next.fit([xs[b] for b in batch], [ys[b] for b in batch], alpha)


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

    def fit(self, xs, ys=None, alpha=None):
        # "xs" contains a nested list with the inputs for a number of instances
        # "ys" contains a nested list with the corresponding correct outcomes
        # "alpha" is the learning rate
        # returns the predictions, losses (if ys not None), and gradients (if alpha not None)
        pass
        return None


class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        # returns an informative description
        text = 'SoftmaxLayer()'
        text += '\n  - next = ' + super().__str__()
        return text

    def fit(self, xs, ys=None, alpha=None):
        # "xs" contains a nested list with the inputs for a number of instances
        # "ys" contains a nested list with the corresponding correct outcomes
        # "alpha" is the learning rate
        # returns the predictions, losses (if ys not None), and gradients (if alpha not None)
        pass
        return None


class LossLayer(Layer):

    def __init__(self, loss_func=quadratic_loss_func):
        # "loss_func" contains a reference to the loss-function
        super().__init__()
        self.loss_func = loss_func
        self.loss_grad = derivative(loss_func)

    def __str__(self):
        # returns an informative description
        text = 'LossLayer():'
        text += '\n  - loss_func = ' + self.loss_func.__name__
        return text

    def fit(self, xs, ys=None, alpha=None):
        # "xs" contains a nested list with the inputs for a number of instances
        # "ys" contains a nested list with the corresponding correct outcomes
        # "alpha" is the learning rate
        # returns the predictions, losses (if ys not None), and gradients (if alpha not None)
        pass
        return None
