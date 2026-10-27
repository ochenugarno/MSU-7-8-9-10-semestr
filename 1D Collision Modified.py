import numpy as np
import matplotlib.pyplot as plt

SIGMA = 50
EPSILON = 1

RS = np.power(26 / 7, 1 / 6) * SIGMA
RC = 67 / 48 * RS
K1 = 387072 / 61009 / RS ** 3
K2 = 24192 / 3211 / RS ** 2

x_1, x_2 = -100., 100.
startVel_1, startVel_2  = 20., -20.

TAU = 0.001
time = np.arange(0, 10, TAU, dtype=np.longdouble)

X_1, X_2 = np.zeros(len(time), dtype=np.longdouble), np.zeros(len(time), dtype=np.longdouble)
vel_1, vel_2 = np.zeros(len(time), dtype=np.longdouble), np.zeros(len(time), dtype=np.longdouble)
acc_1, acc_2 = np.zeros(len(time), dtype=np.longdouble), np.zeros(len(time), dtype=np.longdouble)

X_1[0], X_2[0] = x_1, x_2
vel_1[0], vel_2[0] = startVel_1, startVel_2

r_0 = np.abs(x_1 - x_2)
s_0 = (r_0 / SIGMA) ** 2
if r_0 <= RS:
    acc_1[0] = 24 * EPSILON * (2 / s_0 ** 7 - 1 / s_0 ** 4) * (x_1 - x_2) / SIGMA ** 2
    E_0 = 4 * EPSILON * (1 / s_0 ** 6 - 1 / s_0 ** 3) + (startVel_1 ** 2 + startVel_2 ** 2) / 2
elif RS < r_0 <= RC:
    acc_1[0] = EPSILON * (K1 * 3 * (r_0 - RC) ** 2 / r_0 * (x_1 - x_2) + K2 * 2 * (r_0 - RC) / r_0 * (x_1 - x_2)) / SIGMA ** 2
    E_0 = - EPSILON * (K1 * (r_0 - RC) ** 3 + K2 * (r_0 - RC) ** 2) + (startVel_1 ** 2 + startVel_2 ** 2) / 2
else:
    acc_1[0] = 0
    E_0 = (startVel_1 ** 2 + startVel_2 ** 2) / 2
acc_2[0] = -acc_1[0]

Ediff = np.zeros(len(time), dtype=np.longdouble)


for t in range(1, len(time)):
    X_1[t] = X_1[t - 1] + vel_1[t - 1] * TAU + acc_1[t - 1] * TAU ** 2 / 2
    X_2[t] = X_2[t - 1] + vel_2[t - 1] * TAU + acc_2[t - 1] * TAU ** 2 / 2

    r = np.abs(X_1[t] - X_2[t])
    s = (r / SIGMA) ** 2

    if r <= RS:
        acc_1[t] = 24 * EPSILON * (2 / s ** 7 - 1 / s ** 4) * (X_1[t] - X_2[t]) / SIGMA ** 2
        E = 4 * EPSILON * (1 / s ** 6 - 1 / s ** 3)
    elif RS < r <= RC:
        acc_1[t] = EPSILON * (K1 * 3 * (r - RC) ** 2 / r * (X_1[t] - X_2[t]) + K2 * 2 * (r - RC) / r * (X_1[t] - X_2[t])) / SIGMA ** 2
        E = - EPSILON * (K1 * (r - RC) ** 3 + K2 * (r - RC) ** 2)
    else:
        acc_1[t] = 0
        E = 0
    acc_2[t] = -acc_1[t]

    vel_1[t] = vel_1[t - 1] + (acc_1[t - 1] + acc_1[t]) * TAU / 2
    vel_2[t] = vel_2[t - 1] + (acc_2[t - 1] + acc_2[t]) * TAU / 2

    Ediff[t] = E_0 - E - (vel_1[t] ** 2 + vel_2[t] ** 2) / 2

#plt.plot(time, X_1)
#plt.plot(time, X_2)
plt.plot(time, Ediff)
plt.xlabel('Time')
plt.ylabel('Energe difference')
plt.show()