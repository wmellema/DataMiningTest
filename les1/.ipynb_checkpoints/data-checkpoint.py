#!/usr/bin/python

# Note: no changes need to be made to this file...

# IMPORTS:

from random import gauss
import matplotlib.pyplot as plt


# FUNCTIONS:

def generate_data(num=64, dim=2, bias=None, weights=None, binary=True):
    """Generate a suitable dataset with attributes and outcomes.

    Keyword arguments:
    num     -- the number of instances (default 64)
    dim     -- the dimensionality of the attributes (default 2)
    bias    -- the bias of the model equation (default random)
    weights -- the weights of the model equation (default random)
    binary  -- generate binary classes or continuous values (default True)

    Return values:
    xs      -- the num x dim values of the attributes
    ys      -- the num values of the labels
    """
    # Generate random bias if none provided
    if bias == None:
        bias = round(gauss(0.0, 1.0), 1)
    # Generate random weights if none provided
    if weights == None:
        weights = [round(gauss(0.0, 1.0), 1) for d in range(dim)]
    # Generate attribute data
    xs = [[gauss(0.0, 10.0) for d in range(dim)] for n in range(num)]
    # Generate outcomes
    if binary:
        ys = [-1 if bias+sum(wi*xi for wi, xi in zip(weights, x)) < 0 else 1 for x in xs]
    else:
        ys = [bias+sum(wi*xi for wi, xi in zip(weights, x)) for x in xs]
    # Return values
    return xs, ys


def plot_data(xs, ys, model, final=False):
    """Plots data according to true and modeled values.

    Keyword arguments:
    xs      -- the values of the attributes
    ys      -- the values of the true outcomes
    model   -- the Perceptron or LinearRegression model
    final   -- is this the final figure (default false)
    """
    # Open a new figure and determine color range
    fig, ax = plt.subplots()
    vmin = min(ys)
    vmax = max(ys)
    # Paint background colors denoting the model predictions
    plot_x = [(x/4.0)-32.0 for x in range(257)]
    plot_y = [(y/4.0)-32.0 for y in range(257)]
    plot_z = [[model.predict([x, y]) for x in plot_x] for y in plot_y]
    im = ax.imshow(plot_z, origin='lower', extent=(-32.0, 32.0, -32.0, 32.0), cmap=plt.cm.RdYlBu, vmin=vmin, vmax=vmax)
    # Overlay the actual data
    x0s = [x[0] for x in xs]
    x1s = [x[1] for x in xs]
    plt.scatter(x=x0s, y=x1s, c=ys, edgecolors='w', cmap=plt.cm.RdYlBu, vmin=vmin, vmax=vmax)
    # Finish and show the figure
    fig.colorbar(im)
    ax.axis([-32.0, 32.0, -32.0, 32.0])
    if final:
        plt.show()


# REDIRECT EXECUTION

if __name__ == "__main__":
    print('Warning: data.py is a module; now running script.py instead...')
    import script
