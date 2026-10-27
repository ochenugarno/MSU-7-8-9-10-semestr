import numpy as np
import matplotlib.pyplot as plt

sigma = 50
epsilon = 1

x_1, x_2 = -100., 100.
startVel_1, startVel_2  = 20., -20.

tau = 0.001
time = np.arange(0, 10, tau, dtype=np.longdouble)

X_1, X_2 = np.zeros(len(time), dtype=np.longdouble), np.zeros(len(time), dtype=np.longdouble)
vel_1, vel_2 = np.zeros(len(time), dtype=np.longdouble), np.zeros(len(time), dtype=np.longdouble)
acc_1, acc_2 = np.zeros(len(time), dtype=np.longdouble), np.zeros(len(time), dtype=np.longdouble)

X_1[0], X_2[0] = x_1, x_2
vel_1[0], vel_2[0] = startVel_1, startVel_2

s = ((x_1 - x_2)/sigma)**2
acc_1[0] = 12 * epsilon * (2 / s ** 7 - 1 / s ** 4) * 2 * (x_1 - x_2) / sigma ** 2
acc_2[0] = -acc_1[0]

Ediff = np.zeros(len(time), dtype=np.longdouble)
E_0 = (startVel_1**2 + startVel_2**2)/2 + 4*epsilon*(1/s**6 - 1/s**3)

for i in range(1, len(time)):
    X_1[i] = X_1[i - 1] + vel_1[i - 1] * tau + acc_1[i - 1] * tau ** 2 / 2
    X_2[i] = X_2[i - 1] + vel_2[i - 1] * tau + acc_2[i - 1] * tau ** 2 / 2

    s = ((X_1[i] - X_2[i]) / sigma) ** 2
    acc_1[i] = 12 * epsilon * (2 / s ** 7 - 1 / s ** 4) * 2 * (X_1[i] - X_2[i]) / sigma ** 2
    acc_2[i] = -acc_1[i]

    vel_1[i] = vel_1[i - 1] + (acc_1[i - 1] + acc_1[i])*tau/2
    vel_2[i] = vel_2[i - 1] + (acc_2[i - 1] + acc_2[i]) * tau / 2

    Ediff[i] = E_0 - (vel_1[i]**2 + vel_2[i]**2)/2 - 4*epsilon*(1/s**6 - 1/s**3)
np.savez('data_from_CPU_1D_collision.npz', array1=time, array2=X_1, array3=X_2, array4=Ediff)
plt.plot(time, X_1)
plt.plot(time, X_2)
#plt.plot(time, Ediff)
plt.xlabel('Время')
plt.ylabel('Координата (вычисление на CPU)')
plt.show()