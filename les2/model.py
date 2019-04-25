#!/usr/bin/python

# Note: this is a working solution...

# IMPORTS:

from math import tanh, exp
import sys, model, data


# FUNCTIONS:

# Some activation-functions:

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

# def softsign_act_func(a):
#     return ...

# def logistic_act_func(a):
#     return ...

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

class Neuron():

	def __init__(self, dim=2, act_func=identity_act_func, loss_func=quadratic_loss_func):
		# "dim" equals the dimensionality of the attributes
		# "act_func" contains a reference to the activation function
		# "loss_func" contains a reference to the loss function
		self.bias = 0.0
		self.weights = [0.0 for d in range(dim)]
		self.act_func = act_func
		self.loss_func = loss_func
		
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
		pre_activation = self.bias+sum(wi*xi for wi, xi in zip(self.weights, x))
		post_activation = self.act_func(pre_activation)
		return post_activation
		
	def loss(self, x, y):
		# "x" contains a list with the attributes of a single instance
		y_hat = self.predict(x)
		loss = self.loss_func(y_hat, y)
		# "y" contains the corresponding correct outcome
		return loss
	
	def train(self, x, y, alpha=0):
		# "x" contains a list with the attributes of a single instance
		# "y" contains the corresponding correct outcome
		# "alpha" is the learning rate; choose a suitable default value

		#bereken de helling van de activatie functie
		pre_activation1 = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
		y_hat1 = self.act_func(pre_activation1)
		pre_activation2 = pre_activation1+1e-6
		y_hat2 = self.act_func(pre_activation2)
		gradient_act = (y_hat2-y_hat1)/1e-6

		#bereken de helling van de loss functie
		post_activation1 = self.act_func(pre_activation)
		loss1 = self.loss_func(post_activation1, y)
		post_activation2 = post_activation1 + 1e-6
		loss2 = self.loss_func(post_activation2, y)
		gradient_loss = (loss2-loss1) / 1e-6
		return

	
	def fit(self, xs, ys, alpha=0, epochs=100):
		# "x" contains a nested list with the attributes of multiple instances
		# "y" contains a list with the corresponding correct outcomes
		# "alpha" is the learning rate; choose a suitable default value
		# "epochs" equals the number of epochs
		pass
			
	
# REDIRECT EXECUTION

if __name__ == "__main__":
	print('Warning: model.py is a module; now running script.py instead...')
	import script2
