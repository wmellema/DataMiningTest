# Note: no changes need to be made to this file...

# IMPORTS:

import matplotlib.pyplot as plt
import warnings
from math import sqrt
from random import gauss


# CLASSES

class DummyModel():
    def predict(self, x):
        return 0.0
    def loss(self, x, y):
        return 0.0
    

# FUNCTIONS:

def generate(nominal, noise=0.0, num=64, dim=2, bias=None, weights=None):
    """Generate a suitable dataset with attributes and outcomes.

    Keyword arguments:
    nominal  -- flag indicates nominal classes or continuous values
    noise    -- the amount of noise to add (default 0.0)
    num      -- number of instances (default 64)
    dim      -- dimensionality of the attributes (default 2)
    bias     -- bias of the generating model equation (default random)
    weights  -- weights of the generating model equation (default random)

    Return values:
    xs       -- values of the attributes
    ys       -- values of the labels
    """
    # Generate random bias if none provided
    if bias == None:
        bias = gauss(0.0, 4.0)
    # Generate randomly directed weight vector if none provided
    if weights == None:
        weights = [gauss(0.0, 1.0) for d in range(dim)]
        length = sqrt(sum(wi**2 for wi in weights))
        weights = [wi/length for wi in weights]
    # Generate attribute data
    xs = [[gauss(0.0, 8.0) for d in range(dim)] for n in range(num)]
    # Generate outcomes
    if nominal:
        ys = [-1 if bias+sum(wi*xi for wi, xi in zip(weights, x)) < 0 else 1 for x in xs]
    else:
        ys = [bias+sum(wi*xi for wi, xi in zip(weights, x)) for x in xs]
    # Add noise to the attributes
    xs = [[xs[n][d]+gauss(0.0, noise) for d in range(dim)] for n in range(num)]
    # Return values
    return xs, ys


def graph(*funcs, xmin=-5.0, xmax=5.0):
    """Plots the graph of a given function.

    Keyword arguments:
    funcs    -- one or more functions to be plotted
    xmin     -- the lowest x-value (default -5)
    xmax     -- the highest x-value (default +5)

    Return values:
    None
    """
    # Open a new figure and calculate (x,y)-values
    xs = [xmin+x*(xmax-xmin)/256 for x in range(257)]
    ymin = 0
    ymax = 0
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for n, func in enumerate(funcs):
        ys = [func(x) for x in xs]
        ymin = min(ymin, min(ys))
        ymax = max(ymax, max(ys))
        plt.plot(xs, ys, color=colors[n % len(colors)], linewidth=3.0)
    # Finish the layout and display the figure
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True, color='k', linestyle=':', linewidth=0.5)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=1.0)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.show()    


def plot(xs, ys, model=DummyModel()):
    """Plots data according to true and modeled outcomes.

    Keyword arguments:
    xs       -- the values of the attributes
    ys       -- the values of the true outcomes
    model    -- the generic Neuron model (default none)

    Return values:
    None
    """
    # Open a new figure and determine color range
    fig, ax = plt.subplots()
    v = max(max(ys), -min(ys))
    # Paint background colors denoting the model predictions
    paint_x = [(x/4.0)-30.0 for x in range(241)]
    paint_y = [(y/4.0)-30.0 for y in range(241)]
    paint_z = [[model.predict([xi, yi]) for xi in paint_x] for yi in paint_y]
    im = ax.imshow(paint_z, origin='lower', extent=(-30.0, 30.0, -30.0, 30.0), interpolation='bilinear', cmap=plt.cm.RdYlBu, vmin=-v, vmax=v)
    # Draw dashed line at contour zero, if it exists
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contour(paint_x, paint_y, paint_z, levels=[0.0], colors='k', linestyles='--', linewidths=1.0)
    # Overlay the actual data
    x0s = [x[0] for x in xs]
    x1s = [x[1] for x in xs]
    plt.scatter(x0s, x1s, c=ys, edgecolors='w', cmap=plt.cm.RdYlBu, vmin=-v, vmax=v)
    # Finish the layout and display the figure
    ax.axis([-30.0, 30.0, -30.0, 30.0])
    ax.grid(True, color='k', linestyle=':', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)
    fig.colorbar(im).ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1.0)
    plt.text(-29.0, -29.0, 'Total loss: {:.3f}'.format(sum(model.loss(x, y) for x, y in zip(xs, ys))))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()    
