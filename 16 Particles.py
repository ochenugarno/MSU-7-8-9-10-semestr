import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sigma = 1.
epsilon = 1.
r = np.power(2, 1/6) * sigma

Xstart = np.array([0, r, 2*r, 3*r, 0, r, 2*r, 3*r, 0, r, 2*r, 3*r, 0, r, 2*r, 3*r], dtype=np.longdouble)
Ystart = np.array([0, 0, 0, 0, r, r, r, r, 2*r, 2*r, 2*r, 2*r, 3*r, 3*r, 3*r, 3*r], dtype=np.longdouble)

VXstart = np.zeros(16, dtype=np.longdouble)
VYstart = np.zeros(16, dtype=np.longdouble)

tau = 0.001
time = np.arange(0, 20, tau, dtype=np.longdouble)

X = np.zeros((len(time), 16), dtype=np.longdouble)
Y = np.zeros((len(time), 16), dtype=np.longdouble)

X[0] = Xstart
Y[0] = Ystart

VX = np.zeros((len(time), 16), dtype=np.longdouble)
VY = np.zeros((len(time), 16), dtype=np.longdouble)

VX[0] = VXstart
VY[0] = VYstart

AX = np.zeros((len(time), 16), dtype=np.longdouble)
AY = np.zeros((len(time), 16), dtype=np.longdouble)
for i in range(16):
    for j in range(16):
        if i != j:
            r = np.sqrt((X[0, i] - X[0, j])**2 + (Y[0, i] - Y[0, j])**2)
            s = np.power(r/sigma, 2)
            AX[0, i] += 24 * epsilon * (2/np.power(s, 7) - 1/np.power(s, 4)) * (X[0, i] - X[0, j])/np.power(sigma, 2)
            AY[0, i] += 24 * epsilon * (2/np.power(s, 7) - 1/np.power(s, 4)) * (Y[0, i] - Y[0, j])/np.power(sigma, 2)


#Ediff = np.zeros(len(time), dtype=np.longdouble)
#E_0 = (startVel_1 ** 2 + startVel_2 ** 2) / 2 + 4 * epsilon * (1 / s ** 6 - 1 / s ** 3)

tPush = 100
VPush = 5

for t in range(1, len(time)):
    X[t] = X[t-1] + VX[t-1] * tau + AX[t-1] * tau ** 2 / 2
    Y[t] = Y[t-1] + VY[t-1] * tau + AY[t-1] * tau ** 2 / 2
    for i in range(16):
        for j in range(16):
            r = np.sqrt((X[t, i] - X[t, j])**2 + (Y[t, i] - Y[t, j])**2)
            if i != j:
                s = np.power(r/sigma, 2)
                AX[t, i] += 24 * epsilon * (2/np.power(s, 7) - 1/np.power(s, 4)) * (X[t, i] - X[t, j])/np.power(sigma, 2)
                AY[t, i] += 24 * epsilon * (2/np.power(s, 7) - 1/np.power(s, 4)) * (Y[t, i] - Y[t, j])/np.power(sigma, 2)

        VX[t, i] = VX[t-1, i] + (AX[t-1, i] + AX[t, i]) * tau/2
        VY[t, i] = VY[t-1, i] + (AY[t-1, i] + AY[t, i]) * tau/2 + VPush * (t == tPush and i == 14)

#    Ediff[i] = E_0 - (vel_1[i] ** 2 + vel_2[i] ** 2) / 2 - 4 * epsilon * (1 / s ** 6 - 1 / s ** 3)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-4, 8)
ax.set_ylim(-4, 8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Эксперимент с 16 частицами и теоретическим потенциалом')
ax.grid(True, alpha=0.3)

n_particles = 16
n_frames = 200

particles = ax.scatter([], [], s=100, alpha=0.7)


def animate(frame):
    current_x = X[frame*50, :]
    current_y = Y[frame*50, :]
    current_positions = np.column_stack([current_x, current_y])
    particles.set_offsets(current_positions)
    return [particles]

anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)

plt.tight_layout()
plt.show()