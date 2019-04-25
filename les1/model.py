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


	def predict(self, x):
		# "x" contains a list with the attributes of a single instance
		self.dim = dim
		pre_activation = self.bias
		for d in range(self.dim):
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
		pass
	
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
