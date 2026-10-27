import numpy as np
import matplotlib.pyplot as plt
import random as random
from numba import njit
from matplotlib.animation import FuncAnimation

@njit
def Accel(X, Y, t, i):
    axTotal = 0
    ayTotal = 0
    for j in range(particleNum):
        if i != j:
            r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
            s = np.power(r / SIGMA, 2)
            axTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (X[t, i] - X[t, j]) / np.power(SIGMA, 2)
            ayTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (Y[t, i] - Y[t, j]) / np.power(SIGMA, 2)
    return [axTotal, ayTotal]

@njit
def PotentialEn(X, Y, t):
    epot = 0
    for j in range(particleNum - 1):
        for i in range(j + 1, particleNum):
            r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
            s = np.power(r / SIGMA, 2)
            epot += 4 * EPSILON * (1 / s ** 6 - 1 / s ** 3)
    return epot

@njit
def KineticEn(VX, VY, t):
    V_Sq = np.power(VX[t], 2) + np.power(VY[t], 2)
    ekin = np.sum(V_Sq) / 2
    return ekin



SIGMA = 100.
EPSILON = 150.
R = np.power(2, 1 / 6) * SIGMA
particleNum = 600
TAU = 0.001
T_SHIFT = 1000
SHIFT_LENGTH = 250
ASHIFT = 2000.

initial_data = np.load("Initial_Conditions.npz")
x_start = initial_data["array1"]
y_start = initial_data["array2"]
vx_start = initial_data["array3"]
vy_start = initial_data["array4"]
ax_start = initial_data["array5"]
ay_start = initial_data["array6"]

time = np.arange(0, 10, TAU, dtype=np.float64)

x = np.zeros((len(time), particleNum), dtype=np.float64)
y = np.zeros((len(time), particleNum), dtype=np.float64)
x[0] = x_start
y[0] = y_start

vx = np.zeros((len(time), particleNum), dtype=np.float64)
vy = np.zeros((len(time), particleNum), dtype=np.float64)
vx[0] = vx_start
vy[0] = vy_start

ax = np.zeros((len(time), particleNum), dtype=np.float64)
ay = np.zeros((len(time), particleNum), dtype=np.float64)
ax[0] = vx_start
ay[0] = vy_start


#EPotStart = PotentialEn(X, Y, 0)
#EKinStart = KineticEn(VX, VY, 0)
#EStartTotal = EPotStart + EKinStart
#EDiff = np.zeros(len(time), dtype=np.longdouble)
#Ek = np.zeros(len(time), dtype=np.longdouble)
#Ep = np.zeros(len(time), dtype=np.longdouble)


ax_shift = np.zeros(particleNum, dtype=np.float64)
ay_shift = np.zeros(particleNum, dtype=np.float64)
NumShift = [300, 301, 302, 303,
             320, 321, 322, 323]
Num = np.arange(particleNum)
for i in NumShift:
    np.delete(Num, i)
    ax_shift[i] = ASHIFT
    ay_shift[i] = 0


for t in range(1, len(time)):
    #EPot = 0
    #EKin = 0
    x[t] = x[t - 1] + vx[t - 1] * TAU + ax[t - 1] * TAU ** 2 / 2
    y[t] = y[t - 1] + vy[t - 1] * TAU + ay[t - 1] * TAU ** 2 / 2
    for i in range(particleNum):
        ax[t, i] = Accel(x, y, t, i)[0] + (T_SHIFT <= t <= T_SHIFT + SHIFT_LENGTH) * ax_shift[i]
        ay[t, i] = Accel(x, y, t, i)[1] + (T_SHIFT <= t <= T_SHIFT + SHIFT_LENGTH) * ay_shift[i]

        vx[t, i] = vx[t - 1, i] + (ax[t - 1, i] + ax[t, i]) * TAU / 2
        vy[t, i] = vy[t - 1, i] + (ay[t - 1, i] + ay[t, i]) * TAU / 2
    #EKin = KineticEn(VX, VY, t)
    #EPot = PotentialEn(X, Y, t)
    #EDiff[t] =  EKin + EPot
    #Ep[t] = EPot
    #Ek[t] = EKin



np.savez('data_from_CPU_a=2000.npz', array1=x, array2=y, array3=vx, array4=vy, array5=ax, array6=ay)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2)
n_frames = 200

def animate(frame):
    ax1.clear()

    currentX = x[frame * 50, Num]
    currentY = y[frame * 50, Num]
    currentX_pushed = x[frame * 50, NumShift]
    currentY_pushed = y[frame * 50, NumShift]
    current_positions = np.column_stack([currentX, currentY])
    current_pushed_positions = np.column_stack([currentX_pushed, currentY_pushed])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=40, alpha=0.7, c='cyan', label='particles')
    ax1.scatter(current_pushed_positions[:, 0], current_pushed_positions[:, 1], s=40, alpha=0.7, c='orange')
    ax1.set_xlim(x[0, 0]- R * 2, x[0, 39] + R * 2)
    ax1.set_ylim(y[0, 0]- R * 2, y[0, 599] + R * 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Эксперимент с 600 частицами (CPU)')

#Eafter = [EDiff[TShift + shiftNum + 1]] * (len(time) - TShift - shiftNum - 1)
#ax2.plot(time[TShift + shiftNum + 1::], EDiff[TShift + shiftNum + 1::] - Eafter, 'r-', linewidth=2)

ax2.set_xlabel('Время')
ax2.set_ylabel('Разность энергий')
ax2.set_title('Разность энергии от времени')
anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
plt.tight_layout()
plt.show()