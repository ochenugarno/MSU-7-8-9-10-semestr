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
        r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
        if i != j:
            #аналитика
            # s = np.power(r / sigma, 2)
            # axTotal += 24 * epsilon * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (X[t, i] - X[t, j]) / np.power(sigma, 2)
            # ayTotal += 24 * epsilon * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (Y[t, i] - Y[t, j]) / np.power(sigma, 2)
            #укороченный
           if np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2) <= rc:
               s = r ** 2 / sigma ** 2
               axTotal += 12 * epsilon * (2 / s ** 7 - 1 / s ** 4) * 2 * (X[t, i] - X[t, j]) / sigma ** 2
               ayTotal += 12 * epsilon * (2 / s ** 7 - 1 / s ** 4) * 2 * (Y[t, i] - Y[t, j]) / sigma ** 2
    return [axTotal, ayTotal]




sigma = 60.
epsilon = 55.
r = np.power(2, 1/6) * sigma
particleNum = 600

rs = np.power(26/7, 1/6) * sigma
rc = 67/48 * rs
k1 = 387072/61009/rs**3
k2 = 24192/3211/rs**2

Xstart = np.zeros(particleNum, dtype=np.longdouble)
Ystart = np.zeros(particleNum, dtype=np.longdouble)

VXstart = np.zeros(particleNum, dtype=np.longdouble)
VYstart = np.zeros(particleNum, dtype=np.longdouble)

for i in range(particleNum):
    numInHex = i%40
    if 0 <= numInHex <= 19:
        Xstart[i] = 1/2 * r * (2*(numInHex) + 1)
        Ystart[i] = (i//40) * np.sqrt(3) * r
    elif 19 <= numInHex <= 39:
        Xstart[i] = 1/ 2 * r * (2 * (numInHex - 19))
        Ystart[i] = (i // 40) * np.sqrt(3)* r + r/2*np.sqrt(3)

    VXstart[i] = random.uniform(-1., 1.)
    VYstart[i] = random.uniform(-1., 1.)


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

#EDiff = np.zeros(len(time), dtype=np.longdouble)
#EStart =
for i in range(particleNum):
    AX[0, i] = Accel(X, Y, 0, i)[0]
    AY[0, i] = Accel(X, Y, 0, i)[1]

XShift = np.zeros(particleNum, dtype=np.longdouble)
YShift = np.zeros(particleNum, dtype=np.longdouble)

TShift = 1000
shiftNum = 500

RShift = 2 * r / shiftNum

XShift[300], XShift[301], XShift[302], XShift[303], XShift[304] = (RShift, RShift, RShift, RShift, RShift)
YShift[300], YShift[301], YShift[302], YShift[303], YShift[304] = 0, 0, 0, 0, 0

XShift[320], XShift[321], XShift[322], XShift[323], XShift[324] = RShift, RShift, RShift, RShift, RShift
YShift[320], YShift[321], YShift[322], YShift[323], YShift[324] = 0, 0, 0, 0, 0

XShift[340], XShift[341], XShift[342], XShift[343], XShift[344] = RShift, RShift, RShift, RShift, RShift
YShift[340], YShift[341], YShift[342], YShift[343], YShift[344] = 0, 0, 0, 0, 0

E = np.zeros(len(time), dtype=np.longdouble)

for t in range(1, len(time)):
    X[t] = X[t - 1] + VX[t - 1] * tau + AX[t - 1] * tau**2/2 + (TShift <= t <= TShift + shiftNum) * XShift
    Y[t] = Y[t - 1] + VY[t - 1] * tau + AY[t - 1] * tau**2/2 + (TShift <= t <= TShift + shiftNum) * YShift
    for i in range(particleNum):
        AX[t, i] = Accel(X, Y, t, i)[0]
        AY[t, i] = Accel(X, Y, t, i)[1]

        VX[t, i] = VX[t-1, i] + (AX[t-1, i] + AX[t, i]) * tau/2
        VY[t, i] = VY[t - 1, i] + (AY[t - 1, i] + AY[t, i]) * tau/2
        E[t] += np.sqrt(VX[t, i]**2 + VY[t, i]**2)



#np.savetxt('velocity20', np.sqrt(VX[-1, :]**2 + VY[-1, :]**2))

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2)

#ax1.set_xlim(-35, 80)
#ax1.set_ylim(-35, 80)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Эксперимент со 100 частицами')
ax1.grid(True, alpha=0.3)

n_frames = 200

#particles = ax1.scatter([], [], s=100, alpha=0.7)

def animate(frame):
    ax1.clear()
    #ax2.clear()

    current_data = np.sqrt(VX[frame * 50, :] ** 2 + VY[frame * 50, :] ** 2)
    #x2.hist(current_data, 15, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    current_x = X[frame*50, :]
    current_y = Y[frame*50, :]
    current_positions = np.column_stack([current_x, current_y])
    #particles.set_offsets(current_positions)
    #current_particles = current_positions[frame*50]
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=100, alpha=0.7)
    ax1.set_xlim(X[0, 0]-r*2, X[0, 39]+r*2)
    ax1.set_ylim(Y[0, 0]-r*2, Y[0, 599]+r*2)


    #x = np.linspace(0, 4, 100)
    #y = 3 * x * np.exp(-x ** 2)
    #ax2.plot(x, y, 'r-', linewidth=2)
    #ax2.set_xlim(0, 3.5)
    #ax2.set_ylim(0, 2)
    #ax2.legend()
    #ax2.grid(True, alpha=0.3)
    #ax2.set_aspect('equal')

#ax2.plot()
anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)

plt.tight_layout()
plt.show()