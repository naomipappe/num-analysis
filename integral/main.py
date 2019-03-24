from numpy import sin


def f(x: float, A: float = 1., w: float = 1.):
    return A * x * sin(w * (x ** 2))


if __name__ == '__main__':
    pass
