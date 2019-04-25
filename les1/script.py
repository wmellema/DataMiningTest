#!/usr/bin/python

# Note: uncomment but do not modify the code as you proceed...

# IMPORTS:

from data import generate_data, plot_data
from model import Perceptron, LinearRegression


# MAIN SCRIPT:

# --- Classification ---
# Generates some linearly separable data and applies the perceptron
# Results are plotted after instantiation, one instance, one epoch, and completion
xs, ys = generate_data(binary=True)
myPerceptron = Perceptron()
plot_data(xs, ys, myPerceptron)
# myPerceptron.train(xs[0], ys[0])
# plot_data(xs, ys, myPerceptron)
# myPerceptron.fit(xs, ys, max_epochs=1)
# plot_data(xs, ys, myPerceptron)
# myPerceptron.fit(xs, ys)
plot_data(xs, ys, myPerceptron, final=True)

# # --- Linear Regression ---
# # Generates some linear data and applies linear regression
# # Results are plotted after instantiation, one instance, one epoch, and convergence
xs, ys = generate_data(binary=False)
myLinearRegression = LinearRegression()
plot_data(xs, ys, myLinearRegression)
# myLinearRegression.train(xs[0], ys[0])
# plot_data(xs, ys, myLinearRegression)
# myLinearRegression.fit(xs, ys, max_epochs=1)
# plot_data(xs, ys, myLinearRegression)
myLinearRegression.fit(xs, ys)
plot_data(xs, ys, myLinearRegression, final=True)
print(myLinearRegression)
