import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


class Spline:
    def __init__(self, a: float, b: float, f: Callable[[float], float], nodes: list or None = None):
        self._a = a
        self._b = b
        self.approx = f
        self._nodes = nodes
        self._steps = None
        self._spline = None, 0

    def get_spline(self, n: int = 5):
        if self._spline[0] is not None and self._spline[1] == n:
            return self._spline[0]
        if n <= 1:
            return
        if self._nodes is None:
            self._nodes = np.linspace(self._a, self._b, n + 1)
        self._steps = self._nodes[1:] - self._nodes[:-1]
        matrix, v = self._make_system(n)
        m = np.linalg.solve(matrix, v)

        def spline(x):
            i = np.where(x <= self._nodes)[0]
            i_1 = np.where(x >= self._nodes)[0]
            i = i[0]
            i_1 = i_1[-1]
            x_n = self._nodes[i]
            x_p = self._nodes[i_1]
            return m[i - 1] * ((x_n - x) ** 3) / (6 * self._steps[i - 1]) + \
                m[i] * ((x_n - x) ** 3) / (6 * self._steps[i - 1]) + \
                (self.approx(x_p) - m[i - 1] * (self._steps[i - 1] ** 2) / 6) * (x_n - x) / self._steps[i - 1] + \
                (self.approx(x_n) - m[i] * (self._steps[i - 1] ** 2) / 6) * (x - x_p) / self._steps[i - 1]

        self._spline = spline, n
        return self._spline[0]

    def _make_system(self, n: int):
        matrix = np.zeros((n + 1, n + 1))
        v = np.zeros(n + 1)

        for i in range(n + 1):
            if i == 0:
                matrix[i, i] = 1
            elif i == n:
                matrix[i, i] = 1
            else:
                matrix[i, i - 1] = self._steps[i - 1] / 6
                matrix[i, i] = (self._steps[i - 1] + self._steps[i]) / 3
                matrix[i, i + 1] = self._steps[i] / 6
                v[i] = (self.approx(self._nodes[i + 1]) - self.approx(self._nodes[i])) / self._steps[i] + \
                       (self.approx(self._nodes[i]) - self.approx(self._nodes[i - 1])) / self._steps[i - 1]
        return matrix, v

    def plot_spline(self, n: int = 5):
        nodes = np.linspace(self._a, self._b, 100)
        spline = self.get_spline(n)
        y = [spline(x) for x in nodes]
        plt.plot(nodes, y, color="orange", linestyle='-.', label='Approximation spline')
        plt.plot(nodes, self.approx(nodes), 'k.', label='True function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Spline interpolation")
        plt.legend()
        plt.show()
