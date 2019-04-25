# Note: no changes need to be made to this file...

# IMPORTS:

import matplotlib.pyplot as plt
import warnings, random
from math import pi, cos, sin, sqrt, floor, ceil, atan2
from cmath import phase


# FUNCTIONS:

def scatter(xs, ys, model=None):
    """Plots data according to true and modeled outcomes.

    Keyword arguments:
    xs       -- the values of the attributes
    ys       -- the values of the true outcomes
    model    -- the classification/regression model (default None)

    Return values:
    None
    """
    # Determine the x-range of the data
    x0s = [xi[0] for xi in xs]
    x1s = [xi[1] for xi in xs]
    range_x = ceil(1.1*max(-min(x0s), max(x0s), -min(x1s), max(x1s)))
    paint_x = [(xi/64.0-1.0)*range_x for xi in range(129)]
    # Generate subplots
    axes = len(ys[0])
    fig, axs = plt.subplots(1, axes, figsize=(6.4*axes, 4.8), squeeze=False)
    for n, ax in enumerate(axs[0]):
        # Determine the y-range of the data
        yns = [yi[n] for yi in ys]
        range_y = max(-min(yns), max(yns))
        # Plot the data
        data = ax.scatter(x0s, x1s, c=yns, edgecolors='w', cmap=plt.cm.RdYlBu, vmin=-range_y, vmax=range_y)
        # Paint background colors denoting the model predictions
        if hasattr(model, 'predict'):
            paint_y = [[model.predict([[xi, yi]])[0][n] for xi in paint_x] for yi in paint_x]
            ax.imshow(paint_y, origin='lower', extent=(-range_x, range_x, -range_x, range_x), vmin=-range_y, vmax=range_y, interpolation='bilinear', cmap=plt.cm.RdYlBu)
            # Draw dashed line at contour zero
            with warnings.catch_warnings():   # Ignore warning that zero-contour is absent
                warnings.simplefilter('ignore')
                ax.contour(paint_x, paint_x, paint_y, levels=[0.0], colors='k', linestyles='--', linewidths=1.0)
        else:
            ax.set_facecolor('#F8F8F8')
        # Finish the layout and display the figure
        ax.set_aspect('equal', 'box')
        ax.axis([-range_x, range_x, -range_x, range_x])
        ax.grid(True, color='k', linestyle=':', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1.0)
        ax.set_axisbelow(True)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        cbar = plt.colorbar(data, ax=ax).ax
        cbar.axhline(y=0.5, color='k', linestyle='--', linewidth=1.0)
        cbar.set_title(r'$y$' if axes == 1 else r'$y_{}$'.format(n+1))
    if hasattr(model, 'loss'):
        loss = sum(model.loss(xs, ys))
        plt.suptitle('Total loss: {:.3f}'.format(loss))
    plt.show()


def graph(funcs, *args, xmin=-3.0, xmax=3.0):
    """Plots the graph of a given function.

    Keyword arguments:
    funcs    -- one or more functions to be plotted
    *args    -- extra arguments that should be passed to the function(s) (optional)
    xmin     -- the lowest x-value (default -4.0)
    xmax     -- the highest x-value (default +4.0)

    Return values:
    None
    """
    # Wrap the function in a list, if only one is provided
    if type(funcs) is not list:
        funcs = [funcs]
    # Plot the figures and keep track of their y-range
    xs = [xmin+xi*(xmax-xmin)/256.0 for xi in range(257)]
    ymin = -1.0
    ymax = +1.0
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.subplot(1, 1, 1, facecolor='#F8F8F8')
    for n, func in enumerate(funcs):
        ys = [func(x, *args) for x in xs]
        ymin = min(ymin, floor(min(ys)))
        ymax = max(ymax, ceil(max(ys)))
        plt.plot(xs, ys, color=colors[n % len(colors)], linewidth=3.0, label=func.__code__.co_name)
    # Finish the layout and display the figure
    plt.axis([xmin, xmax, ymin, ymax])
    plt.legend()
    plt.grid(True, color='k', linestyle=':', linewidth=0.5)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=1.0)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.show()


def generate(nominal, num=64, dim=2, bias=None, weights=None, noise=0.0, seed=None):
    """Generate a suitable dataset with attributes and outcomes.

    Keyword arguments:
    nominal  -- flag indicates nominal classes or continuous values
    num      -- number of instances (default 64)
    dim      -- dimensionality of the attributes (default 2)
    bias     -- bias of the generating model equation (default random)
    weights  -- weights of the generating model equation (default random)
    noise    -- the amount of noise to add (default 0.0)
    seed     -- a seed to initialise the random number generator (default random)

    Return values:
    xs       -- values of the attributes
    ys       -- values of the outcomes
    """
    # Seed the random number generator
    random.seed(seed)
    # Generate random bias if none provided
    if bias == None:
        bias = random.gauss(0.0, 4.0)
    # Generate randomly directed weight vector if none provided
    if weights == None:
        weights = [random.gauss(0.0, 1.0) for d in range(dim)]
        length = sqrt(sum(wi**2 for wi in weights))
        weights = [wi/length for wi in weights]
    # Generate attribute data
    xs = [[random.gauss(0.0, 8.0) for d in range(dim)] for n in range(num)]
    # Generate outcomes
    if nominal:
        ys = [-1 if bias+sum(wi*xi for wi, xi in zip(weights, x)) < 0 else 1 for x in xs]
    else:
        ys = [bias+sum(wi*xi for wi, xi in zip(weights, x)) for x in xs]
    # Add noise to the attributes
    xs = [[xs[n][d]+random.gauss(0.0, noise) for d in range(dim)] for n in range(num)]
    # Return values
    return xs, ys


def multinomial(classes, num=512, seed=None):
    """Generate a dataset based on Newton's method applied to 1+(-z)^c=0.

    Keyword arguments:
    classes  -- number of classes to generate
    num      -- number of instances (default 512)
    seed     -- a seed to initialise the random number generator (default random)

    Return values:
    xs       -- values of the attributes x1 and x2
    ys       -- class labels in one-hot encoding
    """
    # Seed the random number generator
    random.seed(seed)
    # Generate attribute data
    rs = [sqrt(0.75*random.random()) for n in range(num)]
    fs = [2.0*pi*random.random() for n in range(num)]
    xs = [[r*cos(f), r*sin(f)] for r, f in zip(rs, fs)]
    # Initialize outcomes
    ys = [[0.0 for c in range(classes)] for n in range(num)]
    # Perform Newton's method
    for n in range(num):
        z_old = -complex(xs[n][0], xs[n][1])
        z_new = (z_old*(classes-1)-z_old**(1-classes))/classes
        while abs(z_new-z_old) > 1e-9:
            z_old = z_new
            z_new = (z_old*(classes-1)-z_old**(1-classes))/classes
        c = int(((phase(-z_new)/pi+1.0)*classes-1.0)/2.0)
        ys[n][c] = 1.0
    # Return values
    return xs, ys
