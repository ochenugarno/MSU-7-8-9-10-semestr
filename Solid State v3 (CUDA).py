import numpy as np
import matplotlib.pyplot as plt
import random as random
from matplotlib.animation import FuncAnimation
from numba import cuda

@cuda.jit
def coord(x, y, vx, vy, ax, ay, x_shift, y_shift, t, TAU, PARTICLE_NUM):
    i = cuda.grid(1)
    if i >= PARTICLE_NUM:
        return
    if TSHIFT <= t <= TSHIFT + SHIFT_LENGTH:
        x[t, i] = x[t - 1, i] + vx[t - 1, i] * TAU + ax[t - 1, i] * TAU ** 2 / 2 + x_shift[i] / SHIFT_LENGTH
        y[t, i] = y[t - 1, i] + vy[t - 1, i] * TAU + ay[t - 1, i] * TAU ** 2 / 2 + y_shift[i] / SHIFT_LENGTH
    else:
        x[t, i] = x[t - 1, i] + vx[t - 1, i] * TAU + ax[t - 1, i] * TAU ** 2 / 2
        y[t, i] = y[t - 1, i] + vy[t - 1, i] * TAU + ay[t - 1, i] * TAU ** 2 / 2

@cuda.jit
def veloc(vx, vy, ax, ay, vx_shift, vy_shift, t, TAU, GAMMA, PARTICLE_NUM):
    i = cuda.grid(1)
    if i >= PARTICLE_NUM:
        return
    if TSHIFT <= t <= TSHIFT + SHIFT_LENGTH:
        vx[t, i] = vx[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (ax[t - 1, i] + ax[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + vx_shift[i]
        vy[t, i] = vy[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (ay[t - 1, i] + ay[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + vy_shift[i]
    else:
        vx[t, i] = vx[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (ax[t - 1, i] + ax[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2)
        vy[t, i] = vy[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (ay[t - 1, i] + ay[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2)


@cuda.jit
def accel(x, y, ax, ay, ax_shift, ay_shift, t, SIGMA, EPSILON, TSHIFT, SHIFT_LENGTH, PARTICLE_NUM):
    i = cuda.grid(1)
    if i >= PARTICLE_NUM:
        return

    ax_total = 0
    ay_total = 0
    for j in range(PARTICLE_NUM):
        if i != j:
            r = ((x[t, i] - x[t, j]) ** 2 + (y[t, i] - y[t, j]) ** 2) ** 0.5
            s = r * r / (SIGMA * SIGMA)
            ax_total += 24 * EPSILON * (2 / s ** 7 - 1 / s ** 4) * (x[t, i] - x[t, j]) / (SIGMA * SIGMA)
            ay_total += 24 * EPSILON * (2 / s ** 7 - 1 / s ** 4) * (y[t, i] - y[t, j]) / (SIGMA * SIGMA)
    if TSHIFT <= t <= TSHIFT + SHIFT_LENGTH:
        ax[t, i] = ax_total + ax_shift[i]
        ay[t, i] = ay_total + ay_shift[i]
    else:
        ax[t, i] = ax_total
        ay[t, i] = ay_total




SIGMA = 100.
EPSILON = 150.
R = np.power(2, 1 / 6) * SIGMA
GAMMA = 0.5
PARTICLE_NUM = 600
ASHIFT = 6000.
VSHIFT = 0.
RSHIFT = 0 * R
TAU = 0.001
END = 10
STEP = END / TAU
TSHIFT = 1000
SHIFT_LENGTH = 250
FRAMES_NUM = 200
DELTA = int(STEP / FRAMES_NUM)

initial_data = np.load("Initial_Conditions.npz")
x_start = initial_data["array1"]
y_start = initial_data["array2"]
vx_start = initial_data["array3"]
vy_start = initial_data["array4"]
ax_start = initial_data["array5"]
ay_start = initial_data["array6"]


time = np.arange(0, END, TAU, dtype=np.float64)

x = np.zeros((len(time), PARTICLE_NUM), dtype=np.float64)
y = np.zeros((len(time), PARTICLE_NUM), dtype=np.float64)
x[0] = x_start
y[0] = y_start

vx = np.zeros((len(time), PARTICLE_NUM), dtype=np.float64)
vy = np.zeros((len(time), PARTICLE_NUM), dtype=np.float64)
vx[0] = vx_start
vy[0] = vy_start

ax = np.zeros((len(time), PARTICLE_NUM), dtype=np.float64)
ay = np.zeros((len(time), PARTICLE_NUM), dtype=np.float64)
ax[0] = ax_start
ay[0] = ay_start

x_shift = np.zeros(PARTICLE_NUM, dtype=np.float64)
y_shift = np.zeros(PARTICLE_NUM, dtype=np.float64)
vx_shift = np.zeros(PARTICLE_NUM, dtype=np.float64)
vy_shift = np.zeros(PARTICLE_NUM, dtype=np.float64)
ax_shift = np.zeros(PARTICLE_NUM, dtype=np.float64)
ay_shift = np.zeros(PARTICLE_NUM, dtype=np.float64)

shift_index = [300, 301, 302, 303,
              320, 321, 322, 323]
# shift_index = [300, 301, 302, 303,
#                320 - 4, 321 - 4, 322 - 4, 323 - 4,
#                340 - 8, 341 - 8, 342 - 8, 343 - 8]
num = np.arange(PARTICLE_NUM)
for i in shift_index:
    np.delete(num, i)
    x_shift[i] = RSHIFT
    y_shift[i] = 0
    vx_shift[i] = VSHIFT
    vy_shift[i] = 0
    ax_shift[i] = ASHIFT
    ay_shift[i] = 0


d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_vx = cuda.to_device(vx)
d_vy = cuda.to_device(vy)
d_ax = cuda.to_device(ax)
d_ay = cuda.to_device(ay)
d_x_shift = cuda.to_device(x_shift)
d_y_shift = cuda.to_device(y_shift)
d_vx_shift = cuda.to_device(vx_shift)
d_vy_shift = cuda.to_device(vy_shift)
d_ax_shift = cuda.to_device(ax_shift)
d_ay_shift = cuda.to_device(ay_shift)

threads = 256
blocks = (PARTICLE_NUM + threads - 1) // threads

for t in range(1, len(time)):
    coord[blocks, threads](d_x, d_y, d_vx, d_vy, d_ax, d_ay, d_x_shift, d_y_shift, t, TAU, PARTICLE_NUM)
    cuda.synchronize()

    accel[blocks, threads](d_x, d_y, d_ax, d_ay, d_ax_shift, d_ay_shift, t, SIGMA, EPSILON, TSHIFT, SHIFT_LENGTH, PARTICLE_NUM)
    cuda.synchronize()

    veloc[blocks, threads](d_vx, d_vy, d_ax, d_ay, d_vx_shift, d_vy_shift, t, TAU, GAMMA, PARTICLE_NUM)
    cuda.synchronize()

x = d_x.copy_to_host()
y = d_y.copy_to_host()
vx = d_vx.copy_to_host()
vy = d_vy.copy_to_host()
ax = d_ax.copy_to_host()
ay = d_ay.copy_to_host()

np.savez('data_from_GPU_a=6000_LONG.npz', array1=x, array2=y, array3=vx, array4=vy, array5=ax, array6=ay)


plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2)


def animate(frame):
    ax1.clear()

    currentX = x[frame * DELTA, num]
    currentY = y[frame * DELTA, num]
    currentX_pushed = x[frame * DELTA, shift_index]
    currentY_pushed = y[frame * DELTA, shift_index]
    current_positions = np.column_stack([currentX, currentY])
    current_pushed_positions = np.column_stack([currentX_pushed, currentY_pushed])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=80, alpha=0.7, c='cyan', label='particles')
    ax1.scatter(current_pushed_positions[:, 0], current_pushed_positions[:, 1], s=80, alpha=0.7, c='orange')
    ax1.set_xlim(x[0, 40]- R * 2, x[0, 39] + R * 2)
    ax1.set_ylim(y[0, 10]- R * 2, y[0, 587] + R * 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Эксперимент с 600 частицами (GPU)')

ax2.set_xlabel('Время')
ax2.set_ylabel('Разность энергий')
ax2.set_title('Разность энергии от времени')
anim = FuncAnimation(fig, animate, frames=FRAMES_NUM, interval=50, blit=False)
plt.tight_layout()
plt.show()