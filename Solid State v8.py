import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.animation import FuncAnimation

@njit
def Wedge_func_minus(x, x0, y0):
    return -np.tan(ANGLE_WEDGE / 2) * x + y0 + x0 * np.tan(ANGLE_WEDGE / 2)

@njit
def Wedge_func_plus(x, x0, y0):
    return np.tan(ANGLE_WEDGE / 2) * x + y0 - x0 * np.tan(ANGLE_WEDGE / 2)

@njit
def Accel(X, Y, t, i):
    axTotal = 0
    ayTotal = 0
    for j in range(PARTICLE_NUM):
        if i != j:
            r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
            s = np.power(r / SIGMA, 2)
            axTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (X[t, i] - X[t, j]) / np.power(SIGMA, 2)
            ayTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (Y[t, i] - Y[t, j]) / np.power(SIGMA, 2)
    return [axTotal, ayTotal]

@njit
def PotentialEn(X, Y, t):
    epot = 0
    for j in range(PARTICLE_NUM - 1):
        for i in range(j + 1, PARTICLE_NUM):
            r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
            s = np.power(r / SIGMA, 2)
            epot += 4 * EPSILON * (1 / s ** 6 - 1 / s ** 3)
    return epot

@njit
def KineticEn(VX, VY, t):
    ekin = 0
    V_Sq = np.power(VX[t], 2) + np.power(VY[t], 2)
    ekin = np.sum(V_Sq) / 2
    return ekin


data = np.load("data_v6.npz")
X_prev = data["array1"]
Y_prev = data["array2"]
VX_prev = data["array3"]
VY_prev = data["array4"]
AX_prev = data["array5"]
AY_prev = data["array6"]

SIGMA = 100.
EPSILON = 150.
GAMMA = 0.5
R = np.power(2, 1 / 6) * SIGMA
PARTICLE_NUM = 600

Disappeared_index = [308, 309, 310, 311]
BRUH = len(Disappeared_index)
PARTICLE_NUM -= BRUH
TAU = 0.001
T_END = 6
STEPS_NUM = T_END / TAU
R_INTERACTION = R / 100
ANGLE_WEDGE = 40 / 180 * np.pi
V_PUSH = 400.
A_MOTION = 5.

time = np.arange(0, T_END, TAU, dtype=np.longdouble)

X = np.zeros((len(time), PARTICLE_NUM), dtype=np.longdouble)
Y = np.zeros((len(time), PARTICLE_NUM), dtype=np.longdouble)
X[0, :] = X_prev[-1, :]
Y[0, :] = Y_prev[-1, :]

VX = np.zeros((len(time), PARTICLE_NUM), dtype=np.longdouble)
VY = np.zeros((len(time), PARTICLE_NUM), dtype=np.longdouble)
VX[0, :] = VX_prev[-1, :]
VY[0, :] = VY_prev[-1, :]

AX = np.zeros((len(time), PARTICLE_NUM), dtype=np.longdouble)
AY = np.zeros((len(time), PARTICLE_NUM), dtype=np.longdouble)
AX[0, :] = AX_prev[-1, :]
AY[0, :] = AY_prev[-1, :]

Lower_index = np.arange(0, 20)
Upper_index = np.arange(PARTICLE_NUM - 20, PARTICLE_NUM)

X_wedge = np.zeros(len(time), dtype=np.longdouble)
Y_wedge = np.zeros(len(time), dtype=np.longdouble)
V_wedge = np.zeros(len(time), dtype=np.longdouble)
X_wedge[0] = -350.
Y_wedge[0] = 1500.
V_wedge[0] = V_PUSH

EPotStart = PotentialEn(X, Y, 0)
EKinStart = KineticEn(VX, VY, 0)
EStartTotal = EPotStart + EKinStart

for i in range(PARTICLE_NUM - BRUH):
    AX[0, i] = Accel(X, Y, 0, i)[0]
    AY[0, i] = Accel(X, Y, 0, i)[1]

EDiff = np.zeros(len(time), dtype=np.longdouble)

Ek = np.zeros(len(time), dtype=np.longdouble)
Ep = np.zeros(len(time), dtype=np.longdouble)


AX_shift = np.zeros(PARTICLE_NUM, dtype=np.longdouble)
AY_shift = np.zeros(PARTICLE_NUM, dtype=np.longdouble)
T_DISAPPEAR = 1000
MOTION_LENGTH = 2000
V_MOTION = 200.

TShift = 0
shiftNum = 3000

for t in range(1, len(X_wedge)):
    Y_wedge[t] = Y_wedge[0]
    if t < TShift:
        X_wedge[t] = X_wedge[0]
        V_wedge[t] = 0
    if TShift <= t <= TShift + shiftNum:
        X_wedge[t] = X_wedge[0] + V_PUSH * TAU * (t - TShift)
        V_wedge[t] = V_PUSH
    elif t > TShift + shiftNum:
        X_wedge[t] = X_wedge[TShift + shiftNum]
        V_wedge[t] = 0





for t in range(1, len(time)):
    EPot = 0
    EKin = 0
    X[t] = X[t - 1] + VX[t - 1] * TAU + AX[t - 1] * TAU ** 2 / 2
    Y[t] = Y[t - 1] + VY[t - 1] * TAU + AY[t - 1] * TAU ** 2 / 2
    for i in range(PARTICLE_NUM):
        if i in Lower_index:
            AX[t, i] = Accel(X, Y, t, i)[0]
            AY[t, i] = Accel(X, Y, t, i)[1] - A_MOTION
        elif i in Upper_index:
            AX[t, i] = Accel(X, Y, t, i)[0]
            AY[t, i] = Accel(X, Y, t, i)[1] + A_MOTION
        else:
            AX[t, i] = Accel(X, Y, t, i)[0]
            AY[t, i] = Accel(X, Y, t, i)[1]

        distance1 = np.abs(np.tan(ANGLE_WEDGE / 2) * X[t, i] + Y[t, i] - Y_wedge[t] - X_wedge[t] * np.tan(ANGLE_WEDGE / 2)) / np.sqrt(1 + np.tan(ANGLE_WEDGE / 2) ** 2)
        distance2 = np.abs(-np.tan(ANGLE_WEDGE / 2) * X[t, i] + Y[t, i] - Y_wedge[t] + X_wedge[t] * np.tan(ANGLE_WEDGE / 2)) / np.sqrt(1 + np.tan(ANGLE_WEDGE / 2) ** 2)

        if distance1 <= distance2 and X[t, i] <= X_wedge[t] and Y[t, i] >= Wedge_func_minus(X[t, i], X_wedge[t], Y_wedge[t]) and distance1 <= R_INTERACTION:
            beta = np.arctan(np.abs(VY[t - 1, i] / (VX[t - 1, i] - V_wedge[t - 1])))
            V = np.sqrt(VY[t - 1, i] ** 2 + (VX[t - 1, i] - V_wedge[t - 1]) ** 2)

            VX_push = -V * np.cos(ANGLE_WEDGE + beta) + V_wedge[t]
            VY_push = V * np.sin(beta + ANGLE_WEDGE)

        elif distance2 < distance1 and X[t, i] <= X_wedge[t] and Y[t, i] <= Wedge_func_plus(X[t, i], X_wedge[t], Y_wedge[t]) and distance2 <= R_INTERACTION:
            beta = np.arctan(np.abs(VY[t - 1, i] / (VX[t - 1, i] - V_wedge[t - 1])))
            V = np.sqrt(VY[t - 1, i] ** 2 + (VX[t - 1, i] - V_wedge[t - 1]) ** 2)

            VX_push = -V * np.cos(ANGLE_WEDGE + beta) + V_wedge[t]
            VY_push = -V * np.sin(beta + ANGLE_WEDGE)

        else:
            VX_push = 0
            VY_push = 0

        VX[t, i] = VX[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AX[t - 1, i] + AX[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + VX_push
        VY[t, i] = VY[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AY[t - 1, i] + AY[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + VY_push
        #VX[t, i] = VX[t - 1, i] + (AX[t - 1, i] + AX[t, i]) * TAU / 2 + VX_push
        #VY[t, i] = VY[t - 1, i] + (AY[t - 1, i] + AY[t, i]) * TAU / 2 + VY_push
    EKin = KineticEn(VX, VY, t)
    EPot = PotentialEn(X, Y, t)
    EDiff[t] =  EKin + EPot
    Ep[t] = EPot
    Ek[t] = EKin


plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2)
FRAMES_NUM = 200
DELTA = int(STEPS_NUM / FRAMES_NUM)

def animate(frame):
    ax1.clear()
    #ax2.clear()
    currentX = X[frame * DELTA, :PARTICLE_NUM]
    currentY = Y[frame * DELTA, :PARTICLE_NUM]
    current_positions = np.column_stack([currentX, currentY])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=40, alpha=0.7, c='cyan', label='particles')
    X_wedge_graph = np.arange(-160, X_wedge[frame * DELTA], 0.1)
    Y1_wedge_graph = -np.tan(ANGLE_WEDGE / 2) * X_wedge_graph + Y_wedge[frame * DELTA] + X_wedge[frame * DELTA] * np.tan(ANGLE_WEDGE / 2)
    Y2_wedge_graph = np.tan(ANGLE_WEDGE / 2) * X_wedge_graph + Y_wedge[frame * DELTA] - X_wedge[frame * DELTA] * np.tan(ANGLE_WEDGE / 2)
    ax1.plot(X_wedge_graph, Y1_wedge_graph, linewidth=1, alpha=1, c='red')
    ax1.plot(X_wedge_graph, Y2_wedge_graph, linewidth=1, alpha=1, c='red')
    ax1.set_xlim(X[0, 40]- R * 2, X[0, 39] + R * 2)
    ax1.set_ylim(Y[0, 10]- R * 2, Y[0, 590] + R * 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Эксперимент с 600 частицами (теоретический потенциал)')

Eafter = [EDiff[TShift + shiftNum + 1]] * (len(time) - TShift - shiftNum - 1)
ax2.plot(time[TShift + shiftNum + 1::], EDiff[TShift + shiftNum + 1::] - Eafter, 'r-', linewidth=2)

#ax2.plot(time[TShift + shiftNum + 1:], -Ep[TShift + shiftNum + 1:] + 110000, 'b-', linewidth=2)
#ax2.plot(time[TShift + shiftNum + 1:], Ek[TShift + shiftNum + 1:], 'r-', linewidth=2)
#ax2.plot(time, EDiff, 'r-', linewidth=2)
ax2.set_xlabel('Время')
ax2.set_ylabel('Разность энергий')
ax2.set_title('Разность энергии от времени')
anim = FuncAnimation(fig, animate, frames=FRAMES_NUM, interval=50, blit=False)
#fig.save('conf1', writer=fig.PillowWriter(fps=30), dpi=100)
plt.tight_layout()
plt.show()

