#!/usr/bin/python

# Note: uncomment the lines with code as you proceed...

# IMPORTS:

from data2 import generate_data, plot_data
from model2 import Neuron, tanh_act_func


# MAIN SCRIPT:

# --- Classification ---
# Generates some nominal data, applies classification, and shows results
xs, ys = generate_data(noise=5.0, nominal=True)
my_neuron = Neuron(act_func=tanh_act_func)
plot_data(xs, ys, my_neuron, title='Model before training', show=True)
# my_neuron.train(xs[0], ys[0])
# plot_data(xs, ys, my_neuron, title='Model after one instance', show=True)
# my_neuron.fit(xs, ys, epochs=1)
# plot_data(xs, ys, my_neuron, title='Model after one epoch', show=True)
# my_neuron.fit(xs, ys)
# plot_data(xs, ys, my_neuron, title='Model after convergence', show=True)
print(my_neuron)

# --- Regression ---
# Generates some continuous data, applies regression, and shows results
xs, ys = generate_data(noise=5.0, nominal=False)
my_neuron = Neuron()
plot_data(xs, ys, my_neuron, title='Model before training', show=True)
# my_neuron.train(xs[0], ys[0])
# plot_data(xs, ys, my_neuron, title='Model after one instance', show=True)
# my_neuron.fit(xs, ys, epochs=1)
# plot_data(xs, ys, my_neuron, title='Model after one epoch', show=True)
# my_neuron.fit(xs, ys)
# plot_data(xs, ys, my_neuron, title='Model after convergence', show=True)
# print(my_neuron)
