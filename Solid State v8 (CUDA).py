import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import cuda, njit
import math as math
from matplotlib.animation import FuncAnimation, PillowWriter

@njit
def Wedge_func_minus(x, x0, y0):
    return -np.tan(ANGLE_WEDGE / 2) * x + y0 + x0 * np.tan(ANGLE_WEDGE / 2)

@njit
def Wedge_func_plus(x, x0, y0):
    return np.tan(ANGLE_WEDGE / 2) * x + y0 - x0 * np.tan(ANGLE_WEDGE / 2)

@cuda.jit
def coord(x, y, vx, vy, ax, ay, t, TAU, PARTICLE_NUM):
    i = cuda.grid(1)
    if i >= PARTICLE_NUM:
        return

    x[t, i] = x[t - 1, i] + vx[t - 1, i] * TAU + ax[t - 1, i] * TAU ** 2 / 2
    y[t, i] = y[t - 1, i] + vy[t - 1, i] * TAU + ay[t - 1, i] * TAU ** 2 / 2

@cuda.jit
def veloc(x, y, vx, vy, ax, ay, x_wedge, y_wedge, v_wedge, t, TAU, GAMMA, R_INTERACTION, ANGLE_WEDGE, PARTICLE_NUM):
    i = cuda.grid(1)
    if i >= PARTICLE_NUM:
        return

    distance1 = math.fabs(math.tan(ANGLE_WEDGE / 2) * x[t, i] + y[t, i] - y_wedge[t] - x_wedge[t] * math.tan(ANGLE_WEDGE / 2)) / math.sqrt(1 + np.tan(ANGLE_WEDGE / 2) ** 2)
    distance2 = math.fabs(-math.tan(ANGLE_WEDGE / 2) * x[t, i] + y[t, i] - y_wedge[t] + x_wedge[t] * math.tan(ANGLE_WEDGE / 2)) / math.sqrt(1 + np.tan(ANGLE_WEDGE / 2) ** 2)

    if distance1 <= distance2 and x[t, i] <= x_wedge[t] and y[t, i] >= Wedge_func_minus(x[t, i], x_wedge[t], y_wedge[t]) and distance1 <= R_INTERACTION:
        beta = math.atan(math.fabs(vy[t - 1, i] / (vx[t - 1, i] - v_wedge[t - 1])))
        V = math.sqrt(vy[t - 1, i] ** 2 + (vx[t - 1, i] - v_wedge[t - 1]) ** 2)

        vx_push = -V * math.cos(ANGLE_WEDGE + beta) + v_wedge[t]
        vy_push = V * math.sin(beta + ANGLE_WEDGE)

    elif distance2 < distance1 and x[t, i] <= x_wedge[t] and y[t, i] <= Wedge_func_plus(x[t, i], x_wedge[t], y_wedge[t]) and distance2 <= R_INTERACTION:
        beta = math.atan(math.fabs(vy[t - 1, i] / (vx[t - 1, i] - v_wedge[t - 1])))
        V = math.sqrt(vy[t - 1, i] ** 2 + (vx[t - 1, i] - v_wedge[t - 1]) ** 2)

        vx_push = -V * math.cos(ANGLE_WEDGE + beta) + v_wedge[t]
        vy_push = -V * math.sin(beta + ANGLE_WEDGE)

    else:
        vx_push = 0
        vy_push = 0

    vx[t, i] = vx[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (ax[t - 1, i] + ax[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + vx_push
    vy[t, i] = vy[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (ay[t - 1, i] + ay[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + vy_push


@cuda.jit
def accel(x, y, ax, ay, t, SIGMA, EPSILON, PARTICLE_NUM):
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
    ax[t, i] = ax_total
    ay[t, i] = ay_total




SIGMA = 100.
EPSILON = 150.
R = np.power(2, 1 / 6) * SIGMA
GAMMA = 0.3
PARTICLE_NUM = 600
TAU = 0.001
END = 5
STEP = END / TAU
TSHIFT = 0
SHIFT_LENGTH = 1500
V_PUSH = 800.
R_INTERACTION = R / 100
ANGLE_WEDGE = 30 / 180 * np.pi

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

x_wedge = np.zeros(len(time), dtype=np.longdouble)
y_wedge = np.zeros(len(time), dtype=np.longdouble)
v_wedge = np.zeros(len(time), dtype=np.longdouble)
x_wedge[0] = -350.
y_wedge[0] = 1500.
v_wedge[0] = V_PUSH

for t in range(1, len(x_wedge)):
    y_wedge[t] = y_wedge[0]
    if t < TSHIFT:
        x_wedge[t] = x_wedge[0]
        v_wedge[t] = 0
    if TSHIFT <= t <= TSHIFT + SHIFT_LENGTH:
        x_wedge[t] = x_wedge[0] + V_PUSH * TAU * (t - TSHIFT)
        v_wedge[t] = V_PUSH
    elif t > TSHIFT + SHIFT_LENGTH:
        x_wedge[t] = x_wedge[TSHIFT + SHIFT_LENGTH]
        v_wedge[t] = 0






d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_vx = cuda.to_device(vx)
d_vy = cuda.to_device(vy)
d_ax = cuda.to_device(ax)
d_ay = cuda.to_device(ay)
d_x_wedge = cuda.to_device(x_wedge)
d_y_wedge = cuda.to_device(y_wedge)
d_v_wedge = cuda.to_device(v_wedge)


threads = 256
blocks = (PARTICLE_NUM + threads - 1) // threads

for t in range(1, len(time)):
    coord[blocks, threads](d_x, d_y, d_vx, d_vy, d_ax, d_ay, t, TAU, PARTICLE_NUM)
    cuda.synchronize()

    accel[blocks, threads](d_x, d_y, d_ax, d_ay, t, SIGMA, EPSILON, PARTICLE_NUM)
    cuda.synchronize()

    veloc[blocks, threads](d_x, d_y, d_vx, d_vy, d_ax, d_ay, d_x_wedge, d_y_wedge, d_v_wedge, t, TAU, GAMMA, R_INTERACTION, ANGLE_WEDGE, PARTICLE_NUM)
    cuda.synchronize()

x = d_x.copy_to_host()
y = d_y.copy_to_host()
vx = d_vx.copy_to_host()
vy = d_vy.copy_to_host()
ax = d_ax.copy_to_host()
ay = d_ay.copy_to_host()


np.savez('data_from_GPU_wedge.npz', array1=x, array2=y, array3=vx, array4=vy, array5=ax, array6=ay)


#plt.style.use('dark_background')
fig, (ax1) = plt.subplots(figsize=(10, 8))
FRAMES_NUM = 200
DELTA = int(STEP / FRAMES_NUM)

def animate(frame):
    ax1.clear()
    currentX = x[frame * DELTA, :PARTICLE_NUM]
    currentY = y[frame * DELTA, :PARTICLE_NUM]
    current_positions = np.column_stack([currentX, currentY])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=80, alpha=0.7, c='blue', label='particles')
    X_wedge_graph = np.arange(-160, x_wedge[frame * DELTA], 0.1)
    Y1_wedge_graph = -np.tan(ANGLE_WEDGE / 2) * X_wedge_graph + y_wedge[frame * DELTA] + x_wedge[frame * DELTA] * np.tan(ANGLE_WEDGE / 2)
    Y2_wedge_graph = np.tan(ANGLE_WEDGE / 2) * X_wedge_graph + y_wedge[frame * DELTA] - x_wedge[frame * DELTA] * np.tan(ANGLE_WEDGE / 2)
    ax1.plot(X_wedge_graph, Y1_wedge_graph, linewidth=1, alpha=1, c='red')
    ax1.plot(X_wedge_graph, Y2_wedge_graph, linewidth=1, alpha=1, c='red')
    ax1.set_xlim(x[0, 40]- R * 2, x[0, 39] + R * 2)
    ax1.set_ylim(y[0, 10]- R * 2, y[0, 590] + R * 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Эксперимент с 600 частицами (GPU) c клиновидным воздействием')


anim = FuncAnimation(fig, animate, frames=FRAMES_NUM, interval=50, blit=False)
writer = PillowWriter(fps=30)
anim.save("воздействие клином с трением.gif", writer=writer, dpi=120)
plt.tight_layout()
plt.show()