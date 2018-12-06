import numpy as np


def interpolate_newton_equidistant(f, x0,
                                   a: float or None=None, b: float or None=None, n: int or None=None,
                                   h: float or None=None) -> float:
    if a is None:
        a = b - h * n
    if b is None:
        b = a + h * n
    if n is None:
        n = (b - a) / h
    if h is None:
        h = (b - a) / n

    t = (x0 - a) / h

    x = equidistant_nodes(a, b, n)

    n = x.shape[0]

    rr = np.zeros((n, n))

    rr[0] = np.vectorize(f)(x)
    for i in range(n - 1):
        rr[i+1, :-1-i] = rr[i, 1:n-i] - rr[i, :-1-i]

    pnx, nk = 0, 1

    for k in range(n):
        pnx, nk = pnx + nk * rr[k, 0], nk * (t - k) / (k + 1)

    return pnx


def interpolate_newton_chebyshew(f, x0, a, b, n):
    x = chebyshev_nodes(a, b, n)
    n = x.shape[0]
    rr = np.zeros((n, n))
    rr[0] = np.vectorize(f)(x)
    for i in range(n - 1):
        rr[i+1, :-1-i] = (rr[i, 1:n-i] - rr[i, :-1-i]) / (x[1+i:] - x[:-1-i])

    pnx = rr[n - 1, 0]

    for i in range(n - 1)[::-1]:
        pnx = pnx * (x0 - x[i]) + rr[i, 0]

    return pnx


def equidistant_nodes(a, b, n):
    h = (b-a)/n
    return np.arange(a, b + h, h)


def chebyshev_nodes(a, b, n):
    x = np.zeros((n+1,))
    for k in range(n+1):
        x[k] = (a+b)/2 - (a-b)/2 * np.cos((2*k+1)*np.pi/(2*(n+1)))
    return x


def omega(x, a, b, n, nodes_gen):
    nodes = np.array(nodes_gen(a, b, n))
    return np.prod([x-nodes[i] for i in range(len(nodes))])


def reverse_Newton(f, y0, a, b, n) -> float:
    h = (b - a) / n

    t = (y0 - a) / h

    x = equidistant_nodes(a, b, n)

    n = x.shape[0]

    rr = np.zeros((n, n))

    rr[0] = np.array(x)
    for i in range(n - 1):
        rr[i+1, :-1-i] = rr[i, 1:n-i] - rr[i, :-1-i]

    pnx, nk = 0, 1

    for k in range(n):
        pnx, nk = pnx + nk * rr[k, 0], nk * (t - k) / (k + 1)
    eps = 0.523
    x_vals = np.linspace(pnx-eps, pnx+eps, 5)
    y_vals = np.vectorize(f)(x_vals)
    n = y_vals.shape[0]
    rr = np.zeros((n, n))
    rr[0] = np.array(x_vals)
    for i in range(n - 1):
        rr[i+1, :-1-i] = (rr[i, 1:n-i] - rr[i, :-1-i]) / \
            (y_vals[1+i:] - y_vals[:-1-i])

    pnx = rr[n - 1, 0]

    for i in range(n - 1)[::-1]:
        pnx = pnx * (y0 - y_vals[i]) + rr[i, 0]
    return pnx
