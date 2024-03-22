from math import sqrt
from scipy.stats import norm
import numpy as np


def brownian(x0, n: int, dt: float, delta: float, out=None):
 
    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def constraints_to_brownian(state, x0=-5, y0=-5, x1=5, y1=5):
    """
    
    """
    x, y = [int(i) for i in state]
    if x < x0:
        x = x0 + (x0 - x)
    elif x  > x1:
        x = x1 - (x - x1)
    if y < y0:
            y = y0 + (y0 - y)
    elif y > y1:
        y = y1 - (y - y1)

    return np.array([[x], [y]])


if __name__ == "main":
    import numpy
    from pylab import plot, show, grid, axis, xlabel, ylabel, title

    # The Wiener process parameter.
    delta = 0.25
    # Total time.
    T = 10.0
    # Number of steps.
    N = 1
    # Time step size
    dt = T/N
    # Initial values of x.
    x = numpy.empty((2,N+1))
    x[:, 0] = 0.0

    brownian(x[:,0], N, dt, delta, out=x[:,1:])

    # Plot the 2D trajectory.
    plot(x[0],x[1])

    # Mark the start and end points.
    plot(x[0,0],x[1,0], 'go')
    plot(x[0,-1], x[1,-1], 'ro')

    # More plot decorations.
    title('2D Brownian Motion')
    xlabel('x', fontsize=16)
    ylabel('y', fontsize=16)
    axis('equal')
    grid(True)
    show()

