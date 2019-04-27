#!/usr/bin/python

# Note: implement the below classes and save this file as "model.py"...

# CLASSES

class Perceptron():

    def __init__(self, dim=2):
        # "dim" equals the dimensionality of the attributes
        self.bias = 1.0
        self.weights = list()
        for d in range(dim):
            self.weights.append(dim)


    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        dim = self.weights
        pre_activation = self.bias
        for d in range(len(dim)):
            pre_activation += self.weights[d] * x[d]
        if pre_activation  < 0.0:
            post_activation = -1.0
        elif pre_activation > 0.0:
            post_activation = 1.0
        else:
            post_activation = 0.0
        return post_activation

    def train(self, x, y):
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct label
        """
        b <- b + (y-y_hat)
        wi <- wi + (y-y_hat)x[i]
        """
        
        self.bias = self.bias + (y - self.predict(x))
        for count,j in enumerate(self.weights):
            self.weights[count] = self.weights[count] + (y - self.predict(x)) * x[count]


    def fit(self, xs, ys, max_epochs=10):
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct labels
        unl = max_epochs is 0
        curr_e = max_epochs
        while True:
            bias = self.bias
            weights = self.weights
            for i,x in enumerate(xs):
                self.train(x,ys[i])
            if bias is self.bias and weights is self.weights:
                break
            if curr_e < 1 and not unl:
                break
            curr_e = curr_e - 1
            
        # "max_epochs" states the maximum allowed number of epochs (0=unlimited)

class LinearRegression():

    def __init__(self, dim=2):
        #"dim" equals the dimensionality of the attributes
        self.bias = 1.0
        self.weights = list()
        for d in range(dim):
            self.weights.append(dim)

    def predict(self, x):
        dim = self.weights
        pre_activation = self.bias
        for d in range(len(dim)):
            pre_activation += self.weights[d] * x[d]
        
        return pre_activation

    def train(self, x, y, alpha=0.01):
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct outcome
        # "alpha" is the learning rate; choose a suitable default value
        self.bias = self.bias + alpha * (y - self.predict(x))
        for count,j in enumerate(self.weights):
            self.weights[count] = self.weights[count] + alpha * (y - self.predict(x)) * x[count]

    def fit(self, xs, ys, alpha=0.01, max_epochs=20):   # Choose a good default value for alpha
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # "max_epochs" states the maximum allowed number of epochs (0=unlimited)
        unl = max_epochs is 0
        curr_e = max_epochs
        while True:
            bias = self.bias
            weights = self.weights
            for i,x in enumerate(xs):
                self.train(x,ys[i], alpha)
            if bias is self.bias and weights is self.weights:
                break
            if curr_e < 1 and not unl:
                break
            curr_e = curr_e - 1

# REDIRECT EXECUTION

if __name__ == "__main__":
    print('Warning: model.py is a module; now running script.py instead...')
    import script
