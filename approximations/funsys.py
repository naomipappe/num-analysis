from typing import Callable

import numpy as np
from scipy.special import eval_legendre

class FunctionSystem:
    def __init__(self):
        self._descr = ' '
        pass

    def generate_system(self, k: int):
        pass

    def get_function(self, k: int):
        pass

    def get_orthogonality_borders(self):
        pass

    @property
    def description(self):
        return self._descr


class ExponentialSystem(FunctionSystem):
    def __init__(self):
        FunctionSystem.__init__(self)
        self.descr = "{exp(kx)} system of functions"

    def __str__(self):
        return "Exponential system"

    def generate_system(self, k: int):
        for i in range(k + 1):
            yield self.get_function(i)

    def get_function(self, k: int):
        return lambda x: np.exp((((k if (k & 1) else -k) + 1) >> 1) * x)

    def get_orthogonality_borders(self):
        return None


class TrigonometricSystem(FunctionSystem):
    def __init__(self):
        FunctionSystem.__init__(self)
        self._descr = "Trigonometric system"

    def __str__(self):
        return self.description

    def generate_system(self, k: int):
        for i in range(k + 1):
            yield self.get_function(i)

    def get_function(self, k: int):
        # return lambda x: (np.sin if (k & 1) else np.cos)(((k + 1) >> 1) * x)
        if k != 0:
            return lambda x: (np.sin if k % 2 == 0 else np.cos)(k * x) / np.sqrt(np.pi)
        return lambda x: 1 / np.sqrt(np.pi * 2)

    def get_orthogonality_borders(self):
        return -np.pi, np.pi


class LegendreSystem(FunctionSystem):

    def __init__(self):
        FunctionSystem.__init__(self)
        self._descr = "Legendre polynom system"
        self._values = dict()
        self._values[0] = lambda x: 1
        self._values[1] = lambda x: x
        self._values[2] = lambda x: 1 / 2 * (3 * x ** 2 - 1)
        self._values[3] = lambda x: 1 / 2 * (5 * x ** 3 - 3 * x)
        self._values[4] = lambda x: 1 / 8 * (35 * x ** 4 - 30 * x ** 2 + 3)
        self._values[5] = lambda x: 1 / 8 * (63 * x ** 5 - 70 * x ** 3 + 15 * x)
        self._values[6] = lambda x: 1 / 16 * (231 * x ** 6 - 315 * x ** 4 + 105 * x**2-5)

    def generate_system(self, k: int):
        for i in range(k + 1):
            yield self.get_function(i)

    def get_function(self, k: int):
        # if k in self._values.keys():
        #     return self._values[k]
        # else:
        #     self._values[k] = lambda x: (2 * k - 1) / k * x * self.get_function(k - 1)(x) - \
        #                  (k - 1) / k * self.get_function(k - 2)(x)
        #     return self._values[k]
        #return lambda x: eval_legendre(k,x)
        return self._values[k]


    def get_orthogonality_borders(self):
        return -1, 1


class PolynomialSystem(FunctionSystem):
    def __init__(self):
        FunctionSystem.__init__(self)
        self._descr = "Polynomial system of functions"

    def generate_system(self, k: int):
        for i in range(k + 1):
            yield self.get_function(i)

    def get_function(self, k: int):
        return lambda x: x ** k

    def get_orthogonality_borders(self):
        return None


def function_rescale(f: Callable[[float], float], old_a: float, old_b: float,
                     new_a: float, new_b: float) -> Callable[[float], float]:
    return lambda x: f((old_b - old_a) / (new_b - new_a) * (x - new_a) + old_a)
