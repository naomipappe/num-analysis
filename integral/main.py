from numpy import inf

import integrate


def f(x):
    return (x ** 2 + 1) / (x ** 4 + 1)


def d2f(x):
    return -2 * x * (x ** 4 + 2 * (x ** 2) - 1) / ((x ** 4 + 1) ** 2)



if __name__ == '__main__':
    integral = integrate.MeanRectangleIntegral(0, inf, f, d2f)
    value = integral.integrate(1e-3, runge=False, adaptive=True)
    print(value)
