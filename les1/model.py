#!/usr/bin/python

# Note: implement the below classes and save this file as "model.py"...

# CLASSES

class Perceptron():

    def __init__(self, dim=2):
        # "dim" equals the dimensionality of the attributes
        self.bias = 0.0
        self.weights = list()
        for d in range(dim):
            self.weights.append(dim)
        print(self.weights)


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
        print(x)
        print(y)
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct label
        y_hat_list = []
        for i in x:
            for j in self.weights:
                print(i)
                print(j)
                summ = float(i) * float(j)
                summm = summ + self.bias
                if summm < 0:
                    y_hat_list.append([-1])
                else:
                    y_hat_list.append([1])
        print(y_hat_list)
        return y_hat_list

    def fit(self, xs, ys, max_epochs=0):
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct labels
        # "max_epochs" states the maximum allowed number of epochs (0=unlimited)
        pass

class LinearRegression():

    def __init__(self, dim=2):
        #"dim" equals the dimensionality of the attributes
        #self.bias = ...
        #self.weights = ...
        pass

    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        #pre_activation = ...
        #return pre_activation
        pass

    def train(self, x, y, alpha=0):
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct outcome
        # "alpha" is the learning rate; choose a suitable default value
        pass

    def fit(self, xs, ys, alpha=0, max_epochs=10):   # Choose a good default value for alpha
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct outcomes
        # "alpha" is the learning rate; choose a suitable default value
        # "max_epochs" states the maximum allowed number of epochs (0=unlimited)
        pass

# REDIRECT EXECUTION

if __name__ == "__main__":
    print('Warning: model.py is a module; now running script.py instead...')
    import script
