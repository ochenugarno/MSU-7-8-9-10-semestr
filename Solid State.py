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
            #аналитика
            s = np.power(r / SIGMA, 2)
            axTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (X[t, i] - X[t, j]) / np.power(SIGMA, 2)
            ayTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (Y[t, i] - Y[t, j]) / np.power(SIGMA, 2)
            #укороченный
           # if np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2) <= rs:
           #     s = r ** 2 / sigma ** 2
           #     axTotal += 12 * epsilon * (2 / s ** 7 - 1 / s ** 4) * 2 * (X[t, i] - X[t, j]) / sigma ** 2
           #     ayTotal += 12 * epsilon * (2 / s ** 7 - 1 / s ** 4) * 2 * (Y[t, i] - Y[t, j]) / sigma ** 2
           # elif rs < np.sqrt((X[0, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2) <= rc:
           #     axTotal += epsilon * (k1 * 3 * (r - rc) ** 2 / r * (X[t, i] - X[t, j]) + k2 * 2 * (r - rc) / r * (X[t, i] - X[t, j]))/sigma**2
           #     ayTotal += epsilon * (k1 * 3 * (r - rc) ** 2 / r * (Y[t, i] - Y[t, j]) + k2 * 2 * (r - rc) / r * (Y[t, i] - Y[t, j]))/sigma**2
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
            #     epot += 4 * EPSILON * (s ** 6 - s ** 3)
            # elif RS < r <= RC:
            #     epot += - EPSILON * (K1 * (r - RC) ** 3 + K2 * (r - RC) ** 2)
            # else:
            #     pot = 0
    return epot

@njit
def KineticEn(VX, VY, t):
    ekin = 0
    V_Sq = np.power(VX[t], 2) + np.power(VY[t], 2)
    ekin = np.sum(V_Sq) / 2
    return ekin

SIGMA = 10.
EPSILON = 50.
R = np.power(2, 1 / 6) * SIGMA
particleNum = 2

rs = np.power(26 / 7, 1 / 6) * SIGMA
rc = 67 / 48 * rs
k1 = 387072 / 61009 / rs ** 3
k2 = 24192 / 3211 / rs ** 2


Xstart = np.zeros(particleNum, dtype=np.longdouble)
Ystart = np.zeros(particleNum, dtype=np.longdouble)

VXstart = np.zeros(particleNum, dtype=np.longdouble)
VYstart = np.zeros(particleNum, dtype=np.longdouble)


# for i in range(particleNum):
#     numInHex = i%22
#     if 0 <= numInHex <= 4:
#         Xstart[i] = np.sqrt(3)/2 * r * (2*(numInHex) + 1)
#         Ystart[i] = (i//22) * 3*r
#     elif 5 <= numInHex <= 10:
#         Xstart[i] = np.sqrt(3) / 2 * r * (2 * (numInHex - 5))
#         Ystart[i] = (i // 22) * 3 * r + r/2
#     elif 11 <= numInHex <= 16:
#         Xstart[i] = np.sqrt(3) / 2 * r * (2 * (numInHex - 11))
#         Ystart[i] = (i // 22) * 3 * r + 3/2*r
#     elif 17 <= numInHex <= 21:
#         Xstart[i] = np.sqrt(3)/2 * r * (2*(numInHex - 17) + 1)
#         Ystart[i] = (i // 22) * 3 * r + 2*r
#
#     VXstart[i] = random.uniform(-2, 2)
#     VYstart[i] = random.uniform(-2, 2)

for i in range(particleNum):
    # numInHex = i%20
    # if 0 <= numInHex <= 9:
    #     Xstart[i] = 1/2 * r * (2*(numInHex) + 1)
    #     Ystart[i] = (i//20) * np.sqrt(3) * r
    # elif 9 <= numInHex <= 19:
    #     Xstart[i] = 1/ 2 * r * (2 * (numInHex - 9))
    #     Ystart[i] = (i // 20) * np.sqrt(3)* r + r/2*np.sqrt(3)

    VXstart[i] = random.uniform(-1., 1.)
    VYstart[i] = random.uniform(-1., 1.)

Xstart[0] = 0
Xstart[1] = R

Ystart[1] = 0
Ystart[1] = 0
tau = 0.001
time = np.arange(0, 10, tau, dtype=np.longdouble)

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


EKinStart = KineticEn(VX, VY, 0)
EPotStart = PotentialEn(X, Y, 0)
for i in range(particleNum):
    AX[0, i] = Accel(X, Y, 0, i)[0]
    AY[0, i] = Accel(X, Y, 0, i)[1]

EStartTotal = EPotStart + EKinStart

EDiff = np.zeros(len(time), dtype=np.longdouble)

XShift = np.zeros(particleNum, dtype=np.longdouble)
YShift = np.zeros(particleNum, dtype=np.longdouble)

TShift = 1000
shiftNum = 500

#RShift = 4 * r / shiftNum
RShift = 0

E1 = np.zeros(len(time), dtype=np.longdouble)
E2 = np.zeros(len(time), dtype=np.longdouble)

for t in range(1, len(time)):
    X[t] = X[t - 1] + VX[t - 1] * tau + AX[t - 1] * tau ** 2 / 2 + (TShift <= t <= TShift + shiftNum) * XShift
    Y[t] = Y[t - 1] + VY[t - 1] * tau + AY[t - 1] * tau ** 2 / 2 + (TShift <= t <= TShift + shiftNum) * YShift
    EPot = 0
    EKin = 0
    for i in range(particleNum):
        AX[t, i] = Accel(X, Y, t, i)[0]
        AY[t, i] = Accel(X, Y, t, i)[1]

        VX[t, i] = VX[t - 1, i] + (AX[t - 1, i] + AX[t, i]) * tau / 2
        VY[t, i] = VY[t - 1, i] + (AY[t - 1, i] + AY[t, i]) * tau / 2
    EPot = PotentialEn(X, Y, t)
    EKin = KineticEn(VX, VY, t)
    E1[t] = EPot
    E2[t] = EKin
    EDiff[t] = EStartTotal - EKin - EPot



#np.savetxt('velocity20', np.sqrt(VX[-1, :]**2 + VY[-1, :]**2))

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Эксперимент со 100 частицами')
ax1.grid(True, alpha=0.3)

n_frames = 500

#particles = ax1.scatter([], [], s=100, alpha=0.7)

def animate(frame):
    ax1.clear()
    #ax2.clear()

    current_data = np.sqrt(VX[frame * 10, :] ** 2 + VY[frame * 10, :] ** 2)
    #x2.hist(current_data, 15, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    current_x = X[frame*10, :]
    current_y = Y[frame*10, :]
    current_positions = np.column_stack([current_x, current_y])
    #particles.set_offsets(current_positions)
    #current_particles = current_positions[frame*50]
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=50, alpha=0.7)
    ax1.set_xlim(-20, 145)
    ax1.set_ylim(-20, 220)


    #x = np.linspace(0, 4, 100)
    #y = 3 * x * np.exp(-x ** 2)
    #ax2.plot(x, y, 'r-', linewidth=2)
    #ax2.set_xlim(0, 3.5)
    #ax2.set_ylim(0, 2)
    #ax2.legend()
    #ax2.grid(True, alpha=0.3)
    #ax2.set_aspect('equal')
print(EStartTotal)
ax2.plot(time, EDiff, 'r-', linewidth=2)
#ax2.plot(time[1:], E[1:], 'r-', linewidth=2)
#ax2.plot(0, EKinStart, 'r-', linewidth=2)
#ax2.plot(time[1:], E2[1:], 'r-', linewidth=2)
#ax2.plot(0, EPotStart, 'b-', linewidth=2)
#ax2.plot(time[1:], E1[1:], 'b-', linewidth=2)
#ax2.plot(time, EDiff, 'r-', linewidth=2)
y = [EStartTotal] * len(time)
#ax2.plot(time, y, 'r-', linewidth=2)

anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)

plt.tight_layout()
plt.show()