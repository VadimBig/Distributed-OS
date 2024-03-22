import matplotlib.pyplot as plt
import numpy as np
from trajectory_equations import eq_circle, eq_partline, eq_sin_or_cos, constraints_to_brownian, brownian

def generate_trajectory(x0, y0, x_start, y_start, x_end, y_end, w, t, direction, way_eq):
    if way_eq == "circle":
        return eq_circle(x0, y0, x_start, y_start, w, t)
    elif way_eq == "partline":
        return eq_partline(x0, y0, x_start, y_start, x_end, y_end, w, t, direction)
    elif way_eq == "sin_or_cos":
        return eq_sin_or_cos(x0, y0, x_start, x_end, y_end, w, t, direction, sin=False)

def generate_brownian_trajectory(x0, y0, x_s, y_s, x_e, y_e, direction, w, n, t):
    return constraints_to_brownian(brownian(x0, y0, n, t, w), x_s, y_s, x_e, y_e)

coord = []
time = []
timestep = 100

x0, y0 = -2, 0
x_start, y_start = -3, 0
x_end, y_end = 3, 0
w = 0.0008
way_eq = "partline"
direction = 1

for t in range(0, 60000, timestep):
    x0, y0, direction = generate_trajectory(x0, y0, x_start, y_start, x_end, y_end, w, timestep, direction, way_eq)
    coord.append((x0, y0))
    time.append(t)

x = [i[0] for i in coord]
y = [i[1] for i in coord]

fig, ax = plt.subplots(3)
fig.set_figheight(15)
fig.set_figwidth(15)

ax[0].plot(time, x)
ax[0].set_title('X')

ax[1].plot(time, y)
ax[1].set_title('Y')

ax[2].scatter(x, y)
ax[2].set_title('X and Y')

plt.show()
