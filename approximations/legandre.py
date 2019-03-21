from typing import Callable
from funsys import function_rescale, LegendreSystem
import numpy as np
from util import dot, dot_discrete


class LegendreApproximation:
    def __init__(self, a: float, b: float, f: Callable[[float], float]):
        self.description = 'Mean quadratic root approximation of function based on Legendre polynoms'
        self.borders = a, b
        self._approx = f
        self._legendre = None
        self.fs = LegendreSystem()

    def get_legendre_approximation(self, n: int = 5, verbose: bool = False):
        approx_borders = self.fs.get_orthogonality_borders()
        self._approx = function_rescale(self._approx, *self.borders, *approx_borders)
        c = np.array([dot(*approx_borders, self._approx,
                          self.fs.get_function(i))*((2 * i + 1) / 2) for i in range(n+1)])
        self._legendre = lambda x: np.dot(c, np.array([self.fs.get_function(i)(x) for i in range(n + 1)]))

        if verbose:
            print("Continuous delta(integral): ", end='')
            self._delta()
            print("Discrete delta: ", end='')
            self._delta_discrete(n)

        self._approx = function_rescale(self._approx, *approx_borders, *self.borders)
        self._legendre = function_rescale(self._legendre, *approx_borders, *self.borders)

        if verbose:
            print("Continuous delta(integral): ", end="")
            self._delta()

        return self._legendre

    def _delta(self):
        print("||f-Qn||^2 =", dot(self._a, self._b, lambda x: self._approx(x) - self._legendre(x),
                                  lambda x: self._approx(x) - self._legendre(x)))

    def _delta_discrete(self, n):
        nodes = np.linspace(*self.borders, n)
        print("||f-Qn||^2 =", dot_discrete(lambda x: self._approx(x)-self._legendre(x),
                                           lambda x: self._approx(x)-self._legendre(x), nodes))

    @property
    def description(self):
        return self._descr

    @description.setter
    def description(self, new_descr: str):
        self._descr = new_descr

    @property
    def borders(self):
        return self._a, self._b

    @borders.setter
    def borders(self, borders: tuple):
        if borders[0] >= borders[1]:
            self._a = borders[1]
            self._b = borders[0]
        else:
            self._a = borders[0]
            self._b = borders[1]

    @property
    def function_system(self):
        if self.fs:
            return self.fs.description
        else:
            return "No function system set"
