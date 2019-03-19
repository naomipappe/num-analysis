import numpy as np
from typing import Callable
from util import square_root_method

class Spline:
    def __init__(self, a: float, b: float, f: Callable[[float], float], rho: float = 1., nodes: list or None = None):
        self.rho = rho
        self._a = a
        self._b = b
        self.approx = f
        self._nodes = nodes
        self._steps = None
        self._spline = None

    def get_spline(self, n: int = 5):
        if n <= 1:
            return
        if self._nodes is None:
            self._nodes = np.linspace(self._a, self._b, n + 1)
        self._steps = self._nodes[1:] - self._nodes[:-1]
        m, mu = self._solve_system(n)

        def spline(x0):
            for i in range(n):
                if self._nodes[i] <= x0 <= self._nodes[i+1]:
                    return m[i] * ((self._nodes[i + 1] - x0) ** 3) / (6 * self._steps[i]) + \
                           m[i + 1] * ((x0 - self._nodes[i]) ** 3) / (6 * self._steps[i]) + \
                           (mu(x0)[i] - m[i] * self._steps[i] ** 2 / 6) * (self._nodes[i + 1] - x0) / self._steps[i] + \
                           (mu(x0)[i + 1] - m[i + 1] * self._steps[i] ** 2 / 6) * (x0 - self._nodes[i]) / self._steps[i]

        self._spline = spline
        return self._spline

    def _solve_system(self, n: int):
        A = np.zeros((n + 1, n + 1))
        H = np.zeros((n + 1, n + 1))

        for i in range(1, n):
            H[i][i - 1] = 1 / self._steps[i - 1]
            H[i][i] = - (1 / self._steps[i - 1] + 1 / self._steps[i])
            H[i][i + 1] = 1 / self._steps[i]

        R = np.diag([self.rho for _ in range(n + 1)])
        b = np.zeros(n + 1)
        for i in range(n + 1):
            if i == 0 or i == n:
                A[i, i] = 1
            else:
                A[i, i - 1] = self._steps[i - 1] / 6
                A[i, i] = (self._steps[i - 1] + self._steps[i]) / 3
                A[i, i + 1] = self._steps[i] / 6
                b[i] = (self.approx(self._nodes[i + 1]) - self.approx(self._nodes[i])) / self._steps[i] \
                    - (self.approx(self._nodes[i]) - self.approx(self._nodes[i - 1])) / self._steps[i - 1]

        T = np.dot(np.linalg.inv(R), H.T)
        T = np.dot(H, T)
        T = A + T
        m = square_root_method(T, b)
        T = np.dot(np.dot(np.linalg.inv(R), H.T), m)
        def mu(x):
            return self.approx(x) - T
        return m, mu
