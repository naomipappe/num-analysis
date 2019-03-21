from math import pi
from typing import Callable

import numpy as np

import funsys as fs
import util
from funsys import function_rescale
from util import dot, square_root_method, dot_discrete


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
        self._a = a
        self._b = b
        self._approx = f
        self.fs = fsystem
        self.mqa = None

    def get_mean_quadratic_approximation(self, n: int = 5, verbose: bool = False):
        approx_borders = self.fs.get_orthogonality_borders()
        if approx_borders is not None:

            self._approx = function_rescale(self._approx, *self.borders, *approx_borders)
        else:
            approx_borders = self.borders

        matrix, v = self._make_system(*approx_borders, n)
        c = square_root_method(matrix, v, verbose=verbose)
        self.mqa = lambda x: np.dot(c, np.array([self.fs.get_function(k)(x) for k in range(n)]))

        if verbose:
            print("Continuous delta(integral): ", end='')
            self._delta()
            print("Discrete delta: ", end='')
            self._delta_discrete(n)

        if approx_borders is not None:
            self._approx = function_rescale(self._approx, *approx_borders, *self.borders)
            self.mqa = function_rescale(self.mqa, *approx_borders, *self.borders)

        if verbose:
            print("Continuous delta(integral): ", end='')
            self._delta()

        return self.mqa

    def _make_system(self, a0, b0, n):
        v = np.array([dot(a0, b0, self._approx, self.fs.get_function(i))
                      for i in range(n)])
        matrix = np.array([[dot(a0, b0, self.fs.get_function(k), self.fs.get_function(j))
                            for j in range(n)] for k in range(n)])
        return matrix, v

    def _delta(self):
        print("||f-Qn||^2 =", util.dot(pi / 2, 3 * pi / 2, lambda x: self._approx(x) - self.mqa(x),
                                       lambda x: self._approx(x) - self.mqa(x)))

    def _delta_discrete(self, n):
        nodes = np.linspace(*self.borders, n)
        print("||f-Qn||^2 =", dot_discrete(lambda x: self._approx(x) - self.mqa(x),
                                           lambda x: self._approx(x) - self.mqa(x), nodes))

    @property
    def description(self):
        return self.descr

    @description.setter
    def description(self, new_descr: str):
        self.descr = new_descr

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

    @function_system.setter
    def function_system(self, new_fs: fs.FunctionSystem):
        self.fs = new_fs




