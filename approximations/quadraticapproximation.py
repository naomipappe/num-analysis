from typing import Callable

import numpy as np

import funsys as fs
from funsys import function_rescale
from util import dot, square_root_method
np.printoptions(20)

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
        self._descr = 'Mean quadratic root approximation of function'
        self._a = a
        self._b = b
        self._approx = f
        self._fs = fsystem
        self._mqa = None

    def get_mean_quadratic_approximation(self, n: int = 5, verbose: bool = False):
        approx_borders = self._fs.get_orthogonality_borders()
        if approx_borders is not None:

            self._approx = function_rescale(self._approx, *self.borders, *approx_borders)
        else:
            approx_borders = self.borders

        matrix, v = self._make_system(*approx_borders, n)
        c = square_root_method(matrix, v, verbose=verbose)
        self._mqa = lambda x: np.dot(c, np.array([self._fs.get_function(k)(x) for k in range(n+1)]))

        if verbose:
            print("Continuous delta(integral): ", end='')
            self._delta(approx_borders)
            if self._fs.get_orthogonality_borders() is not None:
                print("Continuous delta(||f||^2 - sum(c_i^2/||phi_i||^2)): ", end='')
                self._delta_discrete(c, approx_borders)

        if self._fs.get_orthogonality_borders() is not None:
            self._approx = function_rescale(self._approx, *approx_borders, *self.borders)
            self._mqa = function_rescale(self._mqa, *approx_borders, *self.borders)
            print("Continuous delta(integral): ", end='')
            self._delta(self.borders)

        return self._mqa

    def _make_system(self, a0, b0, n):
        v = np.array([dot(a0, b0, self._approx, self._fs.get_function(i))
                      for i in range(n+1)])
        matrix = np.array([[dot(a0, b0, self._fs.get_function(k), self._fs.get_function(j))
                            for j in range(n+1)] for k in range(n+1)])
        return matrix, v

    def _delta(self, borders):
        print("||f-Qn||^2 =", dot(*borders, lambda x: self._approx(x) - self._mqa(x),
                                  lambda x: self._approx(x) - self._mqa(x)))

    def _delta_discrete(self, c, borders):
        yield_sys = [(c[i] ** 2) / dot(*borders, self._fs.get_function(i), self._fs.get_function(i)) for i in
                     range(len(c))]
        print("||f-Qn||^2 =", abs(dot(*borders, self._approx, self._approx) - sum(yield_sys)))

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
        if self._fs:
            return self._fs.description
        else:
            return "No function system set"

    @function_system.setter
    def function_system(self, new_fs: fs.FunctionSystem):
        self._fs = new_fs
