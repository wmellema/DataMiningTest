# Note: implement the below classes...

# CLASSES

class Perceptron():

    def __init__(self, dim=2):
        # "dim" equals the dimensionality of the attributes
        self.bias = 0
        self.weights = 1
        
    def __str__(self):
        # Returns an informative description
        result = 'Perceptron():'
        result += '\n  - bias = ' + str(self.bias)
        result += '\n  - weights = ' + str(self.weights)
        return result

    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        # x invullen in formule: ŷ =sgn(b+∑iwi⋅xi)
        y_hat_list = []
        for i in x:
            for j in i:
#                 print(j)
                summ = j * self.weights
                summm = summ + self.bias
                if summm < 0:
                    y_hat_list.append([1,0])
                else:
                    y_hat_list.append([0,1])
        return y_hat_list
        
    def train(self, x, y):
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct label
        # 𝑏←𝑏+(𝑦−𝑦̂ )   𝑤𝑖←𝑤𝑖+(𝑦−𝑦̂ )𝑥𝑖
        print(y)
        print(self.predict(x))
        print(x)
        #difference_y_and_y_hat = 
        
        pass
    
    def fit(self, xs, ys, epochs=0):
        # "x" contains a nested list with the attributes of multiple instances
        # "y" contains a list with the corresponding correct labels
        # "epochs" equals the number of epochs (0=until finished)
        pass
            
            
class LinearRegression():

    def __init__(self, dim=2):
        # "dim" equals the dimensionality of the attributes
        pass
        
    def __str__(self):
        # Returns an informative description
        result = 'LinearRegression():'
        result += '\n  - bias = ' + str(self.bias)
        result += '\n  - weights = ' + str(self.weights)
        return result

    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
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
