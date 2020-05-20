from typing import Callable

import numpy as np

from funsys import function_rescale, LegendreSystem
from util import dot, plot_approximation


class LegendreApproximation:
    def __init__(self, a: float, b: float, f: Callable[[float], float]):
        self.description = 'Mean quadratic root approximation of function based on Legendre polynoms'
        self.borders = a, b
        self._approx = f
        self._legendre = None
        self.fs = LegendreSystem()

    def get_legendre_approximation(self, n: int = 5, verbose: bool = False):
        approx_borders = self.fs.get_orthogonality_borders()
        self._approx = function_rescale(
            self._approx, *self.borders, *approx_borders)
        c = np.array([dot(*approx_borders, self._approx,
                          self.fs.get_function(i)) * (2 * i + 1) / 2 for i in range(n+1)])
        self._legendre = lambda x: np.dot(c, np.array(
            [self.fs.get_function(i)(x) for i in range(n+1)]))

        if verbose:
            print("Continuous delta(integral): ", end='')
            self._delta(approx_borders)
            print("Continuous delta(||f||^2 - sum(c_i^2*||phi_i||^2)): ", end='')
            self._delta_discrete(c, approx_borders)
            plot_approximation(
                *approx_borders, "Legendre approximation on [-1,1]", self._approx, self._legendre)

        self._approx = function_rescale(
            self._approx, *approx_borders, *self.borders)
        self._legendre = function_rescale(
            self._legendre, *approx_borders, *self.borders)

        if verbose:
            print("Continuous delta(integral): ", end="")
            self._delta(self.borders)

        return self._legendre

    def _delta(self, borders):
        print("||f-Qn||^2 =", dot(*borders, lambda x: self._approx(x) - self._legendre(x),
                                  lambda x: self._approx(x) - self._legendre(x)))

    def _delta_discrete(self, c, borders):
        yield_sys = [(c[i] ** 2) * 2/(2*i+1) for i in
                     range(len(c))]
        print("||f-Qn||^2 =",
              abs(dot(*borders, self._approx, self._approx) - sum(yield_sys)))

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
