# Note: implement the below classes...

# IMPORTS:

from math import tanh, log, exp, sqrt
from copy import deepcopy
from random import uniform


# FUNCTIONS:

## Copy previous definitions of activation-, loss- and derivative-functions

def identity_act_func(a):
    return a

def signum_act_func(a):
    if a > 0:
        return 1.0
    elif a < 0:
        return -1.0
    else:
        return 0.0

def tanh_act_func(a):
    return tanh(a)

# Some loss-functions:

def absolute_loss_func(y_hat, y):
    return abs(y_y_hat)

def quadratic_loss_func(y_hat, y):
    return 0.5*(y-y_hat)**2

#calculating the numeric derivative

def derivative(func, x, *args, **kwargs):
    delta_x = 1e-4
    func_plus = func(x + delta_x, *args, **kwargs)
    func_min = func(x - delta_x, *args, **kwargs)
    return(func_plus - func_min)/(2.0*delta_x)


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
        self.bias = 0.0
        self.weights = list()
        for w in range(inputs):
            self.weights.append(round(uniform((-sqrt(6/(inputs+outputs))), sqrt(6/(inputs+outputs))),3))
        self.act_func = act_func

    def __str__(self):
        # returns an informative description
        text = 'FullLayer():'
        text += '\n  - biases = ' + str(self.bias)
        text += '\n  - weights = ' + str(self.weights)
        text += '\n  - act_func = ' + self.act_func.__name__
        text += '\n  - next = ' + super().__str__()
        return text

    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        # returns the predicted outcome
        #Een fully-connected neurale laag krijgt invoer binnen, berekent hieruit (middels lineaire combinatie en activatiefunctie) diens uitvoer, en geeft die door aan de volgende laag. De volgende laag gaat hier vervolgens de predictie mee vervolgen. De voorspelling die die volgende laag rapporteert wordt tenslotte ook door de fully-connected neurale laag geretourneerd als eigen voorspelling.
        for w in self.weights:
            pre_activation = self.bias + sum(wi*xi for wi, xi in zip(w, x))
        post_activation = self.act_func(pre_activation)
        #Geef post_activation door aan volgende laag
        #volgende laag vervolgt hiermee predictie
        #voorspelling die volgende laag rapporteert wordt door fully-connected neurale laag geretourneerd als eigen voorspelling
        return post_activation

    def loss(self, x, y):
        # "x" contains a list with the attributes of a single instance
        # "y" contains a list with the corresponding correct outcomes
        # returns the loss of the prediction
        y_hat_list = []
        for i in x:
            y_hat_list.append(self.predict(x))
        total_loss = 0
        for y_hat in y_hat_list:
            for i in y:
                total_loss += quadratic_loss_func(y_hat, i)
        return total_loss

    def train(self, x, y, alpha=0.01):
        # "x" contains a list with the attributes of a single instance
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # returns the gradient of the loss with respect to the input x
        pre_activation = [self.biases[j] + sum([x[i] * self.weights[j][i] for i in range(len(x))]) for j in range(self.output)]
        post_activation = [self.act_func(a) for a in pre_activation]
        post_gradient = self.next.train(post_activation, y)
        pre_gradient = [derivative(self.act_func, a) for a in pre_activation]
        input_gradient = [sum([pre_gradient[i] * post_gradient[i] * self.weights[i][j] for i in range(len(pre_gradient))]) for j in range(len(x))]
        return input_gradient

    def fit(self, xs, ys, alpha=0.01, epochs=100):
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # "epochs" equals the number of epochs
        epoch = 0
        while epoch < epochs or epochs == 0:
            error = 0
            for i, x in enumerate(xs):
                self.train(x, ys[i], alpha = alpha)
                error += sum([ys[i][j] - self.predict(x[j]) for j in range(self.outputs)])
            epoch += 1
            if error == 0:
                return None
        


class LossLayer(Layer):

    def __init__(self, loss_func=quadratic_loss_func):
        # "loss_func" contains a reference to the loss-function
        super().__init__()
        self.loss_func=quadratic_loss_func

    def __str__(self):
        # returns an informative description
        text = 'LossLayer():'
        text += '\n  - loss_func = ' + self.loss_func.__name__
        return text

    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        # returns the predicted outcome
        #Een loss-laag krijgt invoer binnen. Die invoer komt al uit de laatste fully-connected neurale laag, dus dat vormt reeds de voorspelling  𝑦̂   van het model. De loss-laag kan dus rechtstreeks de invoer retourneren als voorspelling.
        print(x)
        y_hat = x
        print(x)
        return y_hat

    def loss(self, x, y):
        # "x" contains a list with the attributes of a single instance
        # "y" contains a list with the corresponding correct outcomes
        # returns the loss of the prediction
        y_hat_list = []
        for i in x:
            y_hat_list.append(self.predict(x))
        total_loss = 0
        for y_hat in y_hat_list:
            for i in y:
                total_loss += quadratic_loss_func(y_hat, i)
        return total_loss

    def train(self, x, y, alpha=0):
        # "x" contains a list with the attributes of a single instance
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # returns the gradient of the loss with respect to the input x
        pass
        return None
