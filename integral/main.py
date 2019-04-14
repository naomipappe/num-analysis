from numpy import sin

import integrate


def f(x: float, A: float = 1., w: float = 1.):
    return A * x * sin(w * (x ** 2))


if __name__ == '__main__':
    print(integrate.upper_border_approximation_ihor(1e-5))
