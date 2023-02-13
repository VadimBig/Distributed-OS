import math
from scipy.stats import norm
import numpy as np


def brownian(x, y, n: int, dt: float, delta: float):
    """
    Уравнение броуновского движения:
        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    где N(a,b; t0, t1) нормальная распределённая случайная величина со средним a и
    дисперсией b. Параметры t0 и t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    * x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    * n - число шагов, которые нужно совершить
    * dt - размер шага во времени
    * delta -  определяет скорость броуновского движения
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    * out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """
    x0 = np.asarray([[x], [y]])
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*math.sqrt(dt))
    # If `out` was not given, create an output array.
    out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def constraints_to_brownian(state, x0=-5, y0=-5, x1=5, y1=5):
    """
    * `x0`, `x1` - границы по оси X
    * `y0`, `y1` - границы по оси Y
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

    return x, y, 1


def eq_circle(x0, y0, xc, yc, w, direction, t):
    """
    Уравнение движения по окружности
    * `x0`, `y0` - координаты точки в текущий момент
    * `xc`, `yc` - координаты центра окружности
    * `w` - угловая скорость
    * `t` - момент времени
    * `direction` - направление движение: либо 1, либо -1

    На выход - положение точки в момент времени `t`: `(x, y)`
    """

    phi0 = math.atan(y0 / x0)
    x = ((x0 - xc)**2 + (y0 - yc)**2)**0.5 * math.cos(phi0 + w * t * direction)
    y = ((x0 - xc)**2 + (y0 - yc)**2)**0.5 * math.sin(phi0 + w * t * direction)

    return x, y, direction

def eq_partline(x0, y0, x_start, y_start, x_end, y_end, v, t, direction):
    """
    Уравнение движения вдоль отрезка
    * `x_start`, `y_start` - координаты начала отрезка
    * `x_end`, `y_end` - координаты конца отрезка
    * `x0`, `y0` - текущее положение точки
    * v - скорость точки
    * t - время

    На выход - координаты точки и направление: `(x1, y1, direction)`
    """
    phi = math.asin((y_start - y_end) / (x_start - x_end))
    delta_x = math.cos(phi) * v * t
    delta_y = math.sin(phi) * v * t

    if direction == 1:
        x1 = x0 + delta_x
        y1 = y0 + delta_y
        if x1 > x_end:
            x1 = x_end - (x1 - x_end)
            y1 = y_end - (y1 - y_end)
            direction = -1
    else:
        x1 = x0 - delta_x
        y1 = y0 - delta_y
        if x1 < x_start:
            x1 = x_start + (x_start - x1)
            y1 = y_start + (y_start - y1)
            direction = 1

    return (x1, y1, direction)

def eq_sin_or_cos(x0, y0, x_start, x_end, v, t, direction, sin=True):
    """
    Уравнение движения вдоль синуса/косинуса (неравномерное)
    * `x_start`, `x_end` - координаты начала и конца синуса
    * `x0`, `y0` - текущее положение точки
    * `v` - скорость движения по оси x
    * `t` - время
    * `sin` - булева переменная. Если `True`, то `y1=sin(x1)`, иначе `y1=cos(x1)`

    На выход - координаты точки и направление: `(x1, y1, direction)`
    """
    if direction == 1:
        x1 = x0 + v * t
        if x1 > x_end:
            x1 = x_end - (x1 - x_end)
            direction = -1
    else:
        x1 = x0 - v * t
        if x1 < x_start:
            x1 = x_start + (x_start - x1)
            direction = 1
    
    if sin == True:
        return (x1, math.sin(x1) + y0, direction)
    else:
        return (x1, math.cos(x1) + y0, direction)