import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import table

# coefficients go like this (l, a, d, k, n)
coefficients = (2, -1 / 2, 3, 5, -3)


# rhs of the equation
def f_test(x: float, a: float, k: float, d: float, n: float) -> float:
    return a * np.e ** (a * x) * (a * (a * x + 3) + k * (a * x + 2) * (x * (np.e ** (a * x) + 1) + d) + x * n) + \
        n * np.e ** (a * x) + n


# accurate solution
def u_test(x: np.ndarray, a: float, d: float) -> float:
    return x * (np.e ** (a * x) + 1) + d


def main():
    # init section
    l, a, d, k, n = coefficients
    f = f_test
    u_precise = u_test

    N = 200
    h = l / N
    print('Шаг метода:',h)
    x = np.empty(N + 1)
    u_1 = np.empty(N + 1)
    u_2 = np.empty(N + 1)
    u_3 = np.empty(N + 1)
    u_1_0 = d
    u_2_0 = 2
    u_3_0 = 2*a

    def d_u1(u2): return u2
    def d_u2(u3): return u3
    def d_u3(y, u1, u2, u3): return f(y, a, k, d, n) - 5 * u3 * u1 + 3 * u2

    x[0] = 0
    u_1[0] = u_1_0
    u_2[0] = u_2_0
    u_3[0] = u_3_0

    # algorithm goes here
    for i in range(N):
        k1 = d_u1(u_2[i])
        q1 = d_u2(u_3[i])
        r1 = d_u3(x[i], u_1[i], u_2[i], u_3[i])

        k2 = d_u1(u_2[i] + k1 * h / 2)
        q2 = d_u2(u_3[i] + q1 * h / 2)
        r2 = d_u3(x[i] + h / 2, u_1[i] + k1 * h / 2,
                  u_2[i] + q1 * h / 2, u_3[i] + r1 * h / 2)

        k3 = d_u1(u_2[i] + 0*4/9*h*k1 + 2*k2 * h * 1/18)
        q3 = d_u2(u_3[i] + 0*4/9*h*q1 + 2*q2 * h * 1/18)
        r3 = d_u3(x[i] + h / 2, u_1[i] + 0*4/9*h*k1 + 2*k2 * h * 1/18,
                  u_2[i] + 0*4/9*h*q1 + 2*q2 * h * 1/18, u_3[i] + 0*4/9*h*r1 + 2*r2 * h * 1/18)

        k4 = d_u1(u_2[i] + k3 * h)
        q4 = d_u2(u_3[i] + q3 * h)
        r4 = d_u3(x[i] + h, u_1[i] + k3 * h, u_2[i] + q3 * h, u_3[i] + r3 * h)

        u_1[i + 1] = u_1[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        u_2[i + 1] = u_2[i] + h * (q1 + 2 * q2 + 2 * q3 + q4) / 6
        u_3[i + 1] = u_3[i] + h * (r1 + 2 * r2 + 2 * r3 + r4) / 6
        x[i + 1] = x[i] + h

    # plotting
    plt.plot(x, u_precise(x, a, d), 'r')
    plt.plot(x, u_1, 'b')
    plt.xlim(0, l)
    plt.fill_between(x, u_precise(x, a, d), u_1, color='yellow', alpha='0.5')
    plt.savefig(r'F:\numerical\numerical-analysis-lb\runge_kutta\result.png')
    
    data = np.array((x, u_precise(x, a, d), u_1, abs(u_precise(x, a, d) - u_1))).T
    print(data)


main()
