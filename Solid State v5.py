import numpy as np
import matplotlib.pyplot as plt
import random as random
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
    for j in range(particleNum):
        if i != j:
            r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
            #аналитика
            s = np.power(r / SIGMA, 2)
            axTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (X[t, i] - X[t, j]) / np.power(SIGMA, 2)
            ayTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (Y[t, i] - Y[t, j]) / np.power(SIGMA, 2)
            #укороченный
            # if r <= RS:
            #     s = r ** 2 / SIGMA ** 2
            #     axTotal += 24 * EPSILON * (2 / s ** 7 - 1 / s ** 4) * (X[t, i] - X[t, j]) / SIGMA ** 2
            #     ayTotal += 24 * EPSILON * (2 / s ** 7 - 1 / s ** 4) * (Y[t, i] - Y[t, j]) / SIGMA ** 2
            # elif RS < r <= RC:
            #     axTotal += EPSILON * (K1 * 3 * (r - RC) ** 2 / r * (X[t, i] - X[t, j]) + K2 * 2 * (r - RC) / r * (X[t, i] - X[t, j])) / SIGMA ** 2
            #     ayTotal += EPSILON * (K1 * 3 * (r - RC) ** 2 / r * (Y[t, i] - Y[t, j]) + K2 * 2 * (r - RC) / r * (Y[t, i] - Y[t, j])) / SIGMA ** 2
    return [axTotal, ayTotal]

@njit
def PotentialEn(X, Y, t):
    epot = 0
    for j in range(particleNum - 1):
        for i in range(j + 1, particleNum):
            r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
            # аналитика
            s = np.power(r / SIGMA, 2)
            epot += 4 * EPSILON * (1 / s ** 6 - 1 / s ** 3)
            # укороченный
            # if r <= RS:
            #     s = r ** 2 / SIGMA ** 2
            #     epot += 4 * EPSILON * (1 / s ** 6 - 1 / s ** 3)
            # elif RS < r <= RC:
            #     epot += - EPSILON * (K1 * (r - RC) ** 3 + K2 * (r - RC) ** 2)
    return epot

@njit
def KineticEn(VX, VY, t):
    ekin = 0
    V_Sq = np.power(VX[t], 2) + np.power(VY[t], 2)
    ekin = np.sum(V_Sq) / 2
    return ekin



SIGMA = 100.
EPSILON = 350.
GAMMA = 0.5
APLHA = 1.
R = np.power(2, 1 / 6) * SIGMA
particleNum = 600

RS = np.power(26 / 7, 1 / 6) * SIGMA
RC = 67 / 48 * RS
K1 = 387072 / 61009 / RS ** 3
K2 = 24192 / 3211 / RS ** 2

Xstart = np.zeros(particleNum, dtype=np.longdouble)
Ystart = np.zeros(particleNum, dtype=np.longdouble)

VXstart = np.zeros(particleNum, dtype=np.longdouble)
VYstart = np.zeros(particleNum, dtype=np.longdouble)

for i in range(particleNum):
    numInHex = i % 40
    if 0 <= numInHex <= 19:
        Xstart[i] = 1 / 2 * R * (2 * (numInHex) + 1)
        Ystart[i] = (i // 40) * np.sqrt(3) * R
    elif 19 <= numInHex <= 39:
        Xstart[i] = 1 / 2 * R * (2 * (numInHex - 19))
        Ystart[i] = (i // 40) * np.sqrt(3) * R + R / 2 *np.sqrt(3)
    if 0 <= i <= 19:
        VXstart[i] = 0
        VYstart[i] = 0
    else:
        VXstart[i] = random.uniform(-1., 1.)
        VYstart[i] = random.uniform(-1., 1.)


TAU = 0.001
time = np.arange(0, 10, TAU, dtype=np.longdouble)

X = np.zeros((len(time), particleNum), dtype=np.longdouble)
Y = np.zeros((len(time), particleNum), dtype=np.longdouble)

X[0] = Xstart
Y[0] = Ystart

VX = np.zeros((len(time), particleNum), dtype=np.longdouble)
VY = np.zeros((len(time), particleNum), dtype=np.longdouble)

VX[0] = VXstart
VY[0] = VYstart

AX = np.zeros((len(time), particleNum), dtype=np.longdouble)
AY = np.zeros((len(time), particleNum), dtype=np.longdouble)

X_wedge = np.zeros(len(time), dtype=np.longdouble)
Y_wedge = np.zeros(len(time), dtype=np.longdouble)
V_wedge = np.zeros(len(time), dtype=np.longdouble)
X_wedge[0] = -350.
Y_wedge[0] = 1500.
R_INTERACTION = R / 100
ANGLE_WEDGE = 30 / 180 * np.pi
V_PUSH = 800.
V_wedge[0] = V_PUSH

EPotStart = PotentialEn(X, Y, 0)
EKinStart = KineticEn(VX, VY, 0)
EStartTotal = EPotStart + EKinStart

for i in range(particleNum):
    AX[0, i] = Accel(X, Y, 0, i)[0]
    AY[0, i] = Accel(X, Y, 0, i)[1]

EDiff = np.zeros(len(time), dtype=np.longdouble)

Ek = np.zeros(len(time), dtype=np.longdouble)
Ep = np.zeros(len(time), dtype=np.longdouble)


AX_shift = np.zeros(particleNum, dtype=np.longdouble)
AY_shift = np.zeros(particleNum, dtype=np.longdouble)
TShift = 0
shiftNum = 1500

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


AShift = 4000.
# NumShift = [300, 301, 302, 303,
#             320, 321, 322, 323,
#             340, 341, 342, 343]
NumShift_right = [300, 301, 302, 303]
NumShift_left = [320, 321, 322, 323]
Num = np.arange(particleNum)
for i in NumShift_left:
    np.delete(Num, i)
    AX_shift[i] = AShift * np.sin(15 / 180 * np.pi)
    AY_shift[i] = AShift * np.cos(15 / 180 * np.pi)
for i in NumShift_right:
    np.delete(Num, i)
    AX_shift[i] = 0
    AY_shift[i] = 0

for t in range(1, len(time)):
    EPot = 0
    EKin = 0
    X[t] = X[t - 1] + VX[t - 1] * TAU + AX[t - 1] * TAU ** 2 / 2
    Y[t] = Y[t - 1] + VY[t - 1] * TAU + AY[t - 1] * TAU ** 2 / 2
    for i in range(particleNum):
        AX[t, i] = Accel(X, Y, t, i)[0] * (i >= 20)
        AY[t, i] = Accel(X, Y, t, i)[1] * (i >= 20)

        distance1 = np.abs(np.tan(ANGLE_WEDGE / 2) * X[t, i] + Y[t, i] - Y_wedge[t] - X_wedge[t] * np.tan(ANGLE_WEDGE / 2)) / np.sqrt(1 + np.tan(ANGLE_WEDGE / 2) ** 2)
        distance2 = np.abs(-np.tan(ANGLE_WEDGE / 2) * X[t, i] + Y[t, i] - Y_wedge[t] + X_wedge[t] * np.tan(ANGLE_WEDGE / 2)) / np.sqrt(1 + np.tan(ANGLE_WEDGE / 2) ** 2)

        if distance1 <= distance2 and X[t, i] <= X_wedge[t] and Y[t, i] >= Wedge_func_minus(X[t, i], X_wedge[t], Y_wedge[t]) and distance1 <= R_INTERACTION:
            beta = np.arctan(np.abs(VY[t - 1, i] / (VX[t - 1, i] + V_wedge[t - 1])))
            V = np.sqrt(VY[t - 1, i] ** 2 + (VX[t - 1, i] + V_wedge[t - 1]) ** 2)

            VX_push = V * np.cos(ANGLE_WEDGE + beta)
            VY_push = V * np.sin(beta + ANGLE_WEDGE)

            VX[t, i] = VX[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AX[t - 1, i] + AX[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + VX_push
            VY[t, i] = VY[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AY[t - 1, i] + AY[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + VY_push

        elif distance2 < distance1 and X[t, i] <= X_wedge[t] and Y[t, i] <= Wedge_func_plus(X[t, i], X_wedge[t], Y_wedge[t]) and distance2 <= R_INTERACTION:
            beta = np.arctan(np.abs(VY[t - 1, i] / (VX[t - 1, i] + V_wedge[t - 1])))
            V = np.sqrt(VY[t - 1, i] ** 2 + (VX[t - 1, i] + V_wedge[t - 1]) ** 2)

            VX_push = V * np.cos(ANGLE_WEDGE + beta)
            VY_push = -V * np.sin(beta + ANGLE_WEDGE)

            VX[t, i] = VX[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AX[t - 1, i] + AX[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + VX_push
            VY[t, i] = VY[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AY[t - 1, i] + AY[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2) + VY_push

        else:
            VX[t, i] = VX[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AX[t - 1, i] + AX[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2)
            VY[t, i] = VY[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AY[t - 1, i] + AY[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2)
    EKin = KineticEn(VX, VY, t)
    EPot = PotentialEn(X, Y, t)
    EDiff[t] =  EKin + EPot
    Ep[t] = EPot
    Ek[t] = EKin


plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2)
n_frames = 200

def animate(frame):
    ax1.clear()
    #ax2.clear()
    currentX = X[frame * 50, Num]
    currentY = Y[frame * 50, Num]
    currentX_pushed_left = X[frame * 50, NumShift_left]
    currentY_pushed_left = Y[frame * 50, NumShift_left]
    currentX_pushed_right = X[frame * 50, NumShift_right]
    currentY_pushed_right = Y[frame * 50, NumShift_right]
    current_positions = np.column_stack([currentX, currentY])
    current_pushed_positions_left = np.column_stack([currentX_pushed_left, currentY_pushed_left])
    current_pushed_positions_right = np.column_stack([currentX_pushed_right, currentY_pushed_right])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=40, alpha=0.7, c='cyan', label='particles')
    ax1.scatter(current_pushed_positions_left[:, 0], current_pushed_positions_left[:, 1], s=40, alpha=0.7, c='orange')
    ax1.scatter(current_pushed_positions_right[:, 0], current_pushed_positions_right[:, 1], s=40, alpha=0.7, c='orange')
    X_wedge_graph = np.arange(-160, X_wedge[frame * 50], 0.1)
    Y1_wedge_graph = -np.tan(ANGLE_WEDGE / 2) * X_wedge_graph + Y_wedge[frame * 50] + X_wedge[frame * 50] * np.tan(ANGLE_WEDGE / 2)
    Y2_wedge_graph = np.tan(ANGLE_WEDGE / 2) * X_wedge_graph + Y_wedge[frame * 50] - X_wedge[frame * 50] * np.tan(ANGLE_WEDGE / 2)
    ax1.plot(X_wedge_graph, Y1_wedge_graph, linewidth=1, alpha=1, c='red')
    ax1.plot(X_wedge_graph, Y2_wedge_graph, linewidth=1, alpha=1, c='red')
    ax1.set_xlim(X[0, 0]- R * 2, X[0, 39] + R * 2)
    ax1.set_ylim(Y[0, 0]- R * 2, Y[0, 599] + R * 2)
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
anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
#fig.save('conf1', writer=fig.PillowWriter(fps=30), dpi=100)
plt.tight_layout()
plt.show()