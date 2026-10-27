import numpy as np
import matplotlib.pyplot as plt
import random as random
from numba import njit

@njit
def Accel_start(x, y, i):
    axTotal = 0
    ayTotal = 0
    for j in range(PARTICLE_NUM):
        if i != j:
            r = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            s = np.power(r / SIGMA, 2)
            axTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (x[i] - x[j]) / np.power(SIGMA, 2)
            ayTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (y[i] - y[j]) / np.power(SIGMA, 2)
    return [axTotal, ayTotal]

SIGMA = 100.
EPSILON = 150.
R = np.power(2, 1 / 6) * SIGMA
PARTICLE_NUM = 600

x_start = np.zeros(PARTICLE_NUM, dtype=np.float64)
y_start = np.zeros(PARTICLE_NUM, dtype=np.float64)

vx_start = np.zeros(PARTICLE_NUM, dtype=np.float64)
vy_start = np.zeros(PARTICLE_NUM, dtype=np.float64)

ax_start = np.zeros(PARTICLE_NUM, dtype=np.float64)
ay_start = np.zeros(PARTICLE_NUM, dtype=np.float64)

for i in range(PARTICLE_NUM):
    numInHex = i % 40
    if 0 <= numInHex <= 19:
        x_start[i] = 1 / 2 * R * (2 * (numInHex) + 1)
        y_start[i] = (i // 40) * np.sqrt(3) * R
    elif 19 <= numInHex <= 39:
        x_start[i] = 1 / 2 * R * (2 * (numInHex - 19))
        y_start[i] = (i // 40) * np.sqrt(3) * R + R / 2 *np.sqrt(3)

    vx_start[i] = random.uniform(-1., 1.)
    vy_start[i] = random.uniform(-1., 1.)

for i in range(PARTICLE_NUM):
    ax_start[i] = Accel_start(x_start, y_start, i)[0]
    ay_start[i] = Accel_start(x_start, y_start, i)[1]

np.savez('Initial_Conditions.npz', array1=x_start, array2=y_start, array3=vx_start, array4=vy_start, array5=ax_start, array6=ay_start)