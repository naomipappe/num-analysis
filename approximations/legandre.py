from typing import Callable
from funsys import function_rescale
import numpy as np
import matplotlib.pyplot as plt
from util import dot, dot_discrete

class LegandreApproximation:
    def __init__(self, a: float, b: float, f: Callable[[float], float]):
        self._a = a
        self._b = b
        self._approx = f
        self._legandreC = None
        self._legandreD = None

    def get_legandre_approximation_continuous(self, n: int = 5):
        if self._legandreC is not None and self._legandreC[1] == n:
            return self._legandreC[0]
        a0 = -1
        b0 = 1
        self._approx = function_rescale(self._approx, self._a, self._b, a0, b0)
        c = np.array([dot(a0, b0, self._approx, self._get_legandre_polynom(i))*(2 * i + 1) / 2 for i in range(n+1)])

        def result(x):
            s = 0
            for k in range(n + 1):
                s += c[k] * self._get_legandre_polynom(k)(x)
            return s

        self._legandreC = result, n
        return self._legandreC[0]

    def get_legandre_approximation_discrete(self, n: int = 5, nodes: list or None = None):
        if self._legandreD is not None and self._legandreD[1] == n:
            return self._legandreD[0]
        a0 = -1
        b0 = 1
        self._approx = function_rescale(self._approx, self._a, self._b, a0, b0)

        cost_func_prev = float.fromhex('0x1.fffffffffffffp+1023')  # костыль
        cost_func_new = float.fromhex('0x1.ffffffffffffep+1023')  # костыль

        if nodes is None:
            nodes = np.linspace(a0, b0, n + 1)
        m = 0

        while cost_func_new < cost_func_prev:
            m += 1
            c = np.array([
                dot_discrete(self._approx, self._get_legandre_polynom(i), nodes) * (2 * i + 1) / 2
                for i in range(m + 1)])

            def result(x):
                s = 0
                for k in range(m + 1):
                    s += c[k] * self._get_legandre_polynom(k)(x)
                return s

            def delta(x):
                return result(x) - self._approx(x)

            cost_func_prev, cost_func_new = cost_func_new, dot_discrete(delta, delta, nodes) * n / (n - m)
        m -= 1
        c = c = np.array([
                dot_discrete(self._approx, self._get_legandre_polynom(i), nodes) * (2 * i + 1) / 2
                for i in range(m + 1)])

        def result(x):
            s = 0
            for k in range(m + 1):
                s += c[k] * self._get_legandre_polynom(k)(x)
            return s

        self._legandreD = result, n

        def delta(x):
            return result(x) - self._approx(x)

        cost_func_new = dot_discrete(delta, delta, nodes) * n / (n - m)
        print(f"m = {m}, cost = {cost_func_new}")
        return self._legandreD[0]

    def _get_legandre_polynom(self, k: int):
        if k == 0:
            return lambda x: 1
        if k == 1:
            return lambda x: x
        return lambda x: (2 * k - 1) / k * x * self._get_legandre_polynom(k - 1)(x) - \
                         (k - 1) / k * self._get_legandre_polynom(k - 2)(x)

    def plot_approximation(self, n: int = 5, flag: str = None):
        x = np.linspace(self._a, self._b, 200)
        title = "Legandre continuous approximation"
        if flag == 'c':
            legandre = self.get_legandre_approximation_continuous(n)
        elif flag == 'd':
            title = "Legandre discrete approximation"
            legandre = self.get_legandre_approximation_discrete(n)
        else:
            legandre = self.get_legandre_approximation_continuous(n)
        self._approx = function_rescale(self._approx, -1, 1, self._a, self._b)
        legandre = function_rescale(legandre, -1, 1, self._a, self._b)
        plt.title(title)
        plt.plot(x, legandre(x), color="orange", linestyle='-.', label='Approximation function')
        plt.plot(x, self._approx(x), 'k.', label='True function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
