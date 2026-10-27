import numpy as np
import matplotlib.pyplot as plt
import random as random
from numba import njit
from matplotlib.animation import FuncAnimation


@njit
def Accel(X, Y, t, i):
    axTotal = 0
    ayTotal = 0
    for j in range(PARTICLE_NUM - BRUH):
        if i != j:
            r = np.sqrt((X[t, i] - X[t, j]) ** 2 + (Y[t, i] - Y[t, j]) ** 2)
            s = np.power(r / SIGMA, 2)
            axTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (X[t, i] - X[t, j]) / np.power(SIGMA, 2)
            ayTotal += 24 * EPSILON * (2 / np.power(s, 7) - 1 / np.power(s, 4)) * (Y[t, i] - Y[t, j]) / np.power(SIGMA, 2)
    return [axTotal, ayTotal]

@njit
def PotentialEn(X, Y, t):
    global Disappeared_index
    epot = 0
    for j in range(PARTICLE_NUM - len(Disappeared_index) - 1):
        for i in range(j + 1, PARTICLE_NUM - len(Disappeared_index)):
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



SIGMA = 100.
EPSILON = 150.
GAMMA = 0.5
APLHA = 1.
R = np.power(2, 1 / 6) * SIGMA
PARTICLE_NUM = 600


Xstart = np.zeros(PARTICLE_NUM, dtype=np.longdouble)
Ystart = np.zeros(PARTICLE_NUM, dtype=np.longdouble)

VXstart = np.zeros(PARTICLE_NUM, dtype=np.longdouble)
VYstart = np.zeros(PARTICLE_NUM, dtype=np.longdouble)

Disappeared_index = [308, 309, 310, 311]
Lower_index = np.arange(0, 20)
Upper_index = np.arange(PARTICLE_NUM - len(Disappeared_index) - 20, PARTICLE_NUM - len(Disappeared_index))

for i in range(PARTICLE_NUM):
    numInHex = i % 40
    if 0 <= numInHex <= 19:
        Xstart[i] = 1 / 2 * R * (2 * (numInHex) + 1)
        Ystart[i] = (i // 40) * np.sqrt(3) * R
    elif 19 <= numInHex <= 39:
        Xstart[i] = 1 / 2 * R * (2 * (numInHex - 19))
        Ystart[i] = (i // 40) * np.sqrt(3) * R + R / 2 *np.sqrt(3)

    VXstart[i] = random.uniform(-1., 1.)
    VYstart[i] = random.uniform(-1., 1.)





BRUH = len(Disappeared_index)
TAU = 0.001
T_END = 20
STEPS_NUM = T_END / TAU
time = np.arange(0, T_END, TAU, dtype=np.longdouble)

X = np.zeros((len(time), PARTICLE_NUM - len(Disappeared_index)), dtype=np.longdouble)
Y = np.zeros((len(time), PARTICLE_NUM - len(Disappeared_index)), dtype=np.longdouble)

X[0] = np.delete(Xstart, Disappeared_index)
Y[0] = np.delete(Ystart, Disappeared_index)

VX = np.zeros((len(time), PARTICLE_NUM - len(Disappeared_index)), dtype=np.longdouble)
VY = np.zeros((len(time), PARTICLE_NUM - len(Disappeared_index)), dtype=np.longdouble)

VX[0] = np.delete(VXstart, Disappeared_index)
VY[0] = np.delete(VYstart, Disappeared_index)

AX = np.zeros((len(time), PARTICLE_NUM - len(Disappeared_index)), dtype=np.longdouble)
AY = np.zeros((len(time), PARTICLE_NUM - len(Disappeared_index)), dtype=np.longdouble)


#EPotStart = PotentialEn(X, Y, 0)
#EKinStart = KineticEn(VX, VY, 0)
#EStartTotal = EPotStart + EKinStart

for i in range(PARTICLE_NUM - len(Disappeared_index)):
    AX[0, i] = Accel(X, Y, 0, i)[0]
    AY[0, i] = Accel(X, Y, 0, i)[1]

#EDiff = np.zeros(len(time), dtype=np.longdouble)

#Ek = np.zeros(len(time), dtype=np.longdouble)
#Ep = np.zeros(len(time), dtype=np.longdouble)


AX_shift = np.zeros(PARTICLE_NUM, dtype=np.longdouble)
AY_shift = np.zeros(PARTICLE_NUM, dtype=np.longdouble)
T_DISAPPEAR = 1000
MOTION_LENGTH = 2000
A_MOTION = 5.


Num = np.arange(PARTICLE_NUM)
Num_deleted = Num

np.delete(Num_deleted, Disappeared_index)

for t in range(1, len(time)):
    EPot = 0
    EKin = 0
    X[t] = X[t - 1] + VX[t - 1] * TAU + AX[t - 1] * TAU ** 2 / 2
    Y[t] = Y[t - 1] + VY[t - 1] * TAU + AY[t - 1] * TAU ** 2 / 2
    for i in range(PARTICLE_NUM - BRUH):
        if i in Lower_index:
            AX[t, i] = Accel(X, Y, t, i)[0]
            AY[t, i] = Accel(X, Y, t, i)[1] - A_MOTION
        elif i in Upper_index:
            AX[t, i] = Accel(X, Y, t, i)[0]
            AY[t, i] = Accel(X, Y, t, i)[1] + A_MOTION
        else:
            AX[t, i] = Accel(X, Y, t, i)[0]
            AY[t, i] = Accel(X, Y, t, i)[1]

        # VX[t, i] = VX[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AX[t - 1, i] + AX[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2)
        # VY[t, i] = VY[t - 1, i] * (1 - GAMMA * TAU / 2) / (1 + GAMMA * TAU / 2) + (AY[t - 1, i] + AY[t, i]) * TAU / 2 / (1 + GAMMA * TAU / 2)
        VX[t, i] = VX[t - 1, i] + (AX[t - 1, i] + AX[t, i]) * TAU / 2
        VY[t, i] = VY[t - 1, i] + (AY[t - 1, i] + AY[t, i]) * TAU / 2
        #EKin = KineticEn(VX, VY, t)
        #EPot = PotentialEn(X, Y, t)
        #EDiff[t] =  EKin + EPot
        #Ep[t] = EPot
        #Ek[t] = EKin


np.savez('data_v6.npz', array1=X, array2=Y, array3=VX, array4=VY, array5=AX, array6=AY)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2)
FRAMES_NUM = 100
DELTA = int(STEPS_NUM / FRAMES_NUM)
def animate(frame):
    ax1.clear()
    #ax2.clear()
    currentX = X[frame * DELTA, :PARTICLE_NUM - len(Disappeared_index)]
    currentY = Y[frame * DELTA, :PARTICLE_NUM - len(Disappeared_index)]
    current_positions = np.column_stack([currentX, currentY])
    ax1.scatter(current_positions[:, 0], current_positions[:, 1], s=40, alpha=0.7, c='cyan', label='particles')

    ax1.set_xlim(X[0, 0]- R * 5, X[0, 39] + R * 5)
    ax1.set_ylim(Y[0, 0]- R * 5, Y[0, 599 - len(Disappeared_index)] + R * 5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Растяжение системы с 600 частицами (теоретический потенциал)')

#Eafter = [EDiff[TShift + shiftNum + 1]] * (len(time) - TShift - shiftNum - 1)
#ax2.plot(time[TShift + shiftNum + 1::], EDiff[TShift + shiftNum + 1::] - Eafter, 'r-', linewidth=2)

#ax2.plot(time[TShift + shiftNum + 1:], -Ep[TShift + shiftNum + 1:] + 110000, 'b-', linewidth=2)
#ax2.plot(time[TShift + shiftNum + 1:], Ek[TShift + shiftNum + 1:], 'r-', linewidth=2)
#ax2.plot(time, EDiff, 'r-', linewidth=2)
#ax2.set_xlabel('Время')
#ax2.set_ylabel('Разность энергий')
#ax2.set_title('Разность энергии от времени')
anim = FuncAnimation(fig, animate, frames=FRAMES_NUM, interval=50, blit=False)
#fig.save('conf1', writer=fig.PillowWriter(fps=30), dpi=100)
plt.tight_layout()
plt.show()