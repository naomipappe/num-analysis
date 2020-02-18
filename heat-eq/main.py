from collections.abc import Iterable
from typing import Callable

import numpy as np
import tabulate as tb
from matplotlib import pyplot as plt


def plotter(nodes: list, approximation: Callable[[float], float] or list, save: bool = False, name: str = "result"):
    if isinstance(approximation, Iterable):
        approx = approximation
    else:
        approx = [approximation(node) for node in nodes]
    plt.plot(nodes, approx, "b")
    plt.ylim(0, 150)
    if save:
        plt.savefig(f"{name}.png")
    else:
        plt.show()


KELVIN_CELCIUS_DELTA = 273.15
# region task specific constants and functions
m = 2
R = 0.3
lambd = 45.5
cc = 0.46 * (10 ** 3)
rho = 790
gamma = 140
temp_env = 20 + KELVIN_CELCIUS_DELTA
T = 20*60


def initial_temp(x: float) -> float:
    return 100 + 20 * np.sin(x * (R - x)) + KELVIN_CELCIUS_DELTA


# endregion


# region shapeless constants and function
a_squared = lambd / (cc * rho)
T1 = a_squared * T / (R ** 2)
gamma1 = gamma * R / (lambd)


def initial_temp_shapeless(x: float) -> float:
    return (initial_temp(x) - temp_env) / temp_env


# endregion


def convert_temperature(temperature: float) -> float:
    return temperature*temp_env+temp_env-KELVIN_CELCIUS_DELTA


def print_converted(temperatures):
    print([convert_temperature(temp) for temp in temperatures])


alpha1 = 1
alpha2 = 1
beta1 = mu1 = 0
beta2 = gamma1
mu2 = 0

h = 0.1
tau = 0.01
print('Tau =', tau)
print('h =', h)
N = int(np.ceil(1 / h))
M = int(np.ceil(T1 / tau))
sigma = 0.5

x = [i * h for i in range(N + 1)]

dx = np.zeros_like(x)
dx[0] = (x[1] ** (m + 1) - x[0] ** (m + 1)) / ((m + 1) * h)
dx[N] = (x[N] ** (m + 1) - x[N - 1] ** (m + 1)) / ((m + 1) * h)
for i in range(1, N):
    dx[i] = (x[i] ** (m + 1) - x[i - 1] ** (m + 1)) / ((m + 1) * 2 * h)

dp = np.zeros(N)
for i in range(len(dp)):
    dp[i] = (x[i + 1] - h / 2) ** 2

y = [initial_temp_shapeless(point) for point in x]
table = [[0] + [convert_temperature(temp) for temp in y]]
time = 0
center = list()
center.append(convert_temperature(y[0]))
time_moments = list()
time_moments.append(time)
for j in range(M):
    time += tau
    # region construct a scheme
    A = np.zeros((N + 1, N + 1))
    yy = np.zeros(N + 1)

    b = tau * sigma / (h ** 2) * dp[0]
    c = - b - dx[0] / 2
    phi = -(1 - sigma) * tau / (h ** 2) * dp[0] * (y[1] - y[0]) - dx[0] * y[0] / 2
    A[0][0] = c
    A[0][1] = b
    yy[0] = phi

    for i in range(1, N):
        b = tau * sigma / (h ** 2) * dp[i]
        d = tau * sigma / (h ** 2) * dp[i - 1]
        c = -dx[i] - (d + b)
        phi = -dx[i] * y[i] - tau * (1 - sigma) / (h ** 2) * \
              (dp[i] * (y[i + 1] - y[i]) - dp[i - 1] * (y[i] - y[i - 1]))
        A[i][i - 1] = d
        A[i][i] = c
        A[i][i + 1] = b
        yy[i] = phi

    d = tau * sigma / (h ** 2) * dp[N - 1]
    c = -d - sigma * tau / h * x[N] * beta2 - dx[N] / 2
    phi = (1 - sigma) * tau / (h ** 2) * dp[N - 1] * (y[N] - y[N - 1]) + tau / \
          h * (1 - sigma) * x[N] * beta2 * y[N] - tau / h * x[N] * mu2 - dx[N] * y[N] / 2

    A[N][N - 1] = d
    A[N][N] = c
    yy[N] = phi

    y = np.linalg.solve(A, yy)
    center.append(convert_temperature(y[0]))
    time_moments.append(time * (R ** 2) / a_squared / 60)
    table.append([time * (R ** 2) / a_squared / 60] +
                 [convert_temperature(temp) for temp in y])
    # endregion

head = ["Time"] + ["{0:.2f}".format(point * R) for point in x]
with open(r'.\heat.txt', '+w') as outfile:
    for line in tb.tabulate(table, headers=head):
        outfile.write(line)

print("Распределение температуры доступно в M моментов времени, M =", M)
splits = int(input('Показывать распределение температуры в  каждый n-тый момент времени при n = '))
for i in range(0, M, splits):
    plt.plot(x, table[i][1:])
plt.show()
