# Note: implement the below classes...

# CLASSES

class Perceptron():

    def __init__(self, dim=2):
        # "dim" equals the dimensionality of the attributes
        # The number of dim means how many input layers there are so you need to initalize the bias and weights similarly
        # 
        self.bias = 0.0
        self.weights = [0.0 for i in range(dim)]
        
        
        
    def __str__(self):
        # Returns an informative description
        result = 'Perceptron():'
        result += '\n  - bias = ' + str(self.bias)
        result += '\n  - weights = ' + str(self.weights)
        return result
    
    def squashing_function(self,value):
        # This is a binary step function
        if value >= 0:
            return 1
        else :
            return -1
    
    def predict(self, x):
        # "x" contains a list with the attributes of a single instance
        # x invullen in formule: ŷ =sgn(b+∑iwi⋅xi)
        y_hat = 0
        
        output = [0,0]
        
        for j in range(len(self.weights)):
            activation = 0.0
            for i in range(len(x)):
                activation = activation + self.weights[j] * x[i][j]
            
            end_activation = self.bias + activation
            y_hat = self.squashing_function(end_activation)
            output[j] = y_hat
        
        return [output for i in x]
        
    def train(self, x, y):
        # "x" contains a list with the attributes of a single instance
        # "y" contains the corresponding correct label
        # 𝑏←𝑏+(𝑦−𝑦̂ )   𝑤𝑖←𝑤𝑖+(𝑦−𝑦̂ )𝑥𝑖
        print(y)
        print(x)
        
        for i in range(len(y)):
            # Calculate error i.e (𝑦−𝑦̂ )
            for j in range(len(x[0])):
                error = y[i][j] - self.predict(x)[i][j]
            
                # Train the bias
            
                self.bias = self.bias + error
            
                # Train the weights
            
                self.weights[j] = self.weights[j] + error * x[i][j]
            
    
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
