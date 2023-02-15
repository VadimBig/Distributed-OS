import matplotlib.pyplot as plt
import numpy as np
from trajectory_equations import eq_circle, eq_partline, eq_sin_or_cos, constraints_to_brownian, brownian

# circle CHECK
# x0 = 1
# y0 = 1
# xc = 0
# yc = 0
# power = 10
# w = 0.0008 # 3 км/ч
# way_eq = "circle"
# direction = 1

# f = lambda x0, y0, d, t: eq_circle(x0, y0, xc, yc, w, d, t)


# partline y-0 - (-2, -2, 2, 2) до w=0.005 CHECK
x0 = -2
y0 = 0
x_start= -3
y_start = -0
x_end = 3
y_end = 0
power= 1,
w = 0.0008
way_eq = "partline"
direction= 1

f = lambda x0, y0, d, t: eq_partline(x0, y0, x_start, y_start, x_end, y_end, w, t, d)

# partline y-grow - (-2, -2, 2, 2) до w=0.005 CHECK
# x0 = 0
# y0 = 0
# x_start= -2
# y_start = -2
# x_end = 2
# y_end = 2
# power= 1,
# w = 0.0008
# way_eq = "partline"
# direction= 1

# f = lambda x0, y0, d, t: eq_partline(x0, y0, x_start, y_start, x_end, y_end, w, t, d)

# partline y-fall
# x0 = 0
# y0 = 0
# x_start= -2
# y_start = 2
# x_end = 2
# y_end = -2
# power= 1,
# w = 0.0008
# way_eq = "partline"
# direction= 1

# f = lambda x0, y0, d, t: eq_partline(x0, y0, x_start, y_start, x_end, y_end, w, t, d)

# sin_cos - CHECK
# x0 = 0
# y0 = 2
# y = 2
# x_start= -4
# x_end = 4
# power= 5,
# v = 0.0008
# way_eq = "sin_or_cos"
# direction= 1

# f = lambda x0, y0, d, t: eq_sin_or_cos(x0, y0, x_start, x_end, y, v, t, direction, sin=False)

# brownian
# x0, y0, x_s, y_s, x_e, y_e = 2, 6, -10, -10, 10, 10
# direction, w = 1, 0.5
# n = 1

# f = lambda x0, y0, d, t: constraints_to_brownian(brownian(x0, y0, n, t, w), x_s, y_s, x_e, y_e)

coord = []
time = []

timestep = 100

for t in range(0, 60000, timestep): # 60000
    # print(f(x0, y0, direction, t))
    x0, y0, direction = f(x0, y0, direction, timestep)
    coord.append((x0, y0))
    time.append(t)

x = [i[0] for i in coord]
y = [i[1] for i in coord]

print(coord)

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