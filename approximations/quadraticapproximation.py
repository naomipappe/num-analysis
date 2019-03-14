import funsys as fs
import numpy as np
from math import pi
from typing import Callable
import matplotlib.pyplot as plt
from funsys import function_rescale
from util import dot, dot_discrete, square_root_method


class QuadraticApproximation:
    def __init__(self, a: float or None = None,
                 b: float or None = None,
                 f: Callable[[float], float] or None = None,
                 fsystem: fs.FunctionSystem or None = None):
        """
        :type a: left border
        :type b: right border
        :type f: function to approximate
        :type f: approximation system of functions
        """
        self.descr = 'Mean quadratic root approximation of function'
        self.a = a
        self.b = b
        self.approx_function = f
        self.fsystem = fsystem
        self.mgaC = None
        self.mgaD = None

    def get_mean_quadratic_approximation(self, n: int = 5):
        if self.mgaC:
            if self.mgaC[1] == n:
                return self.mgaC[0]
        if isinstance(self.fsystem, fs.TrigonometricSystem):
            a0, b0 = 0, 2 * np.pi
            self.approx_function = function_rescale(
                self.approx_function, *self.borders, a0, b0)
        else:
            a0, b0 = self.borders
        matrix, v = self._make_system_continuous(a0, b0, n)
        c = np.linalg.solve(matrix, v)
        self.mgaC = lambda x: np.dot(c, np.array([self.fsystem.get_function(k)(x) for k in range(n+1)])), n
        return self.mgaC[0]

    def _make_system_continuous(self, a0, b0, n):
        v = np.array([dot(a0, b0, self.approx_function, self.fsystem.get_function(i))
                      for i in range(n+1)])
        matrix = np.array([[dot(a0, b0, self.fsystem.get_function(k), self.fsystem.get_function(j))
                            for j in range(n+1)] for k in range(n+1)])
        return matrix, v

    def get_mean_quadratic_approximation_discrete(self, n: int = 5, nodes: list or None = None):
        if self.mgaD is not None:
            if self.mgaD[1] == n:
                return self.mgaD[0]
        if nodes is None:
            nodes = np.linspace(self.a, self.b, n + 1)
        # TO:DO refactor this function to make it more readable
        cost_func_prev = float.fromhex('0x1.fffffffffffffp+1023')
        cost_func_new = float.fromhex('0x1.ffffffffffffep+1023')

        m = 0
        while cost_func_new < cost_func_prev:
            m += 1
            matrix, v = self._make_system_discrete(m, nodes)
            c = np.linalg.solve(matrix, v)

            def polynom(x):
                s = 0
                x_pow = 1
                for k in range(m + 1):
                    s += c[k] * x_pow
                    x_pow *= x
                return s

            def delta(x):
                return self.approx_function(x) - polynom(x)

            cost_func_prev, cost_func_new = cost_func_new, dot_discrete(delta, delta, nodes) * n / (n - m)
            # true value of m
        m -= 1

        matrix, v = self._make_system_discrete(m, nodes)

        c = np.linalg.solve(matrix, v)

        def polynom(x):
                s = 0
                x_pow = 1
                for k in range(m + 1):
                    s += c[k] * x_pow
                    x_pow *= x
                return s

        self.mgaD = polynom, n

        def diff(x):
            return self.approx_function(x) - polynom(x)

        cost_func_new = dot_discrete(diff, diff, nodes) * n / (n - m)

        print(f'm = {m}, cost = {cost_func_new}')

        return self.mgaD[0]

    def _make_system_discrete(self, m, nodes):
        v = np.array([
            dot_discrete(self.approx_function, lambda x: x**i, nodes)
            for i in range(m + 1)])
        matrix = np.array([[dot_discrete(lambda x: x**j, lambda x: x**i, nodes)
                            for j in range(m + 1)]
                           for i in range(m + 1)])
        return matrix, v

    def plot_approximation(self, n: int = 5, flag: str = None):
        x = np.linspace(self.a, self.b, 200)
        title = f"{str(self.fsystem)} continuous approximation"
        if flag == 'c':
            title = f"{str(self.fsystem)} continuous approximation"
            mga = self.get_mean_quadratic_approximation(n)
        elif flag == 'd':
            title = f"Polynomial discrete approximation"
            mga = self.get_mean_quadratic_approximation_discrete(n)
        else:
            mga = self.get_mean_quadratic_approximation(n)
        if isinstance(self.fsystem, fs.TrigonometricSystem) and flag == 'c':
            self.approx_function = function_rescale(self.approx_function, 0, 2 * pi, *self.borders)
            mga = function_rescale(mga, 0, 2 * pi, *self.borders)
        plt.title(title)
        plt.plot(x, mga(x), color="orange", linestyle='-.', label='Approximation function')
        plt.plot(x, self.approx_function(x), 'k.', label='True function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    @property
    def description(self):
        return self.descr

    @description.setter
    def description(self, new_descr: str):
        self.descr = new_descr

    @property
    def borders(self):
        return self.a, self.b

    @borders.setter
    def borders(self, borders: tuple):
        if borders[0] >= borders[1]:
            self.a = borders[1]
            self.b = borders[0]
        else:
            self.a = borders[0]
            self.b = borders[1]

    @property
    def function_system(self):
        if self.fsystem:
            return self.fsystem.description
        else:
            return "No function system set"

    @function_system.setter
    def function_system(self, new_fs: fs.FunctionSystem):
        self.fsystem = new_fs