from math import pi
from typing import Callable

import numpy as np
from numpy import linspace
from numpy.core._multiarray_umath import pi

import funsys as fs
import util
from funsys import function_rescale
from util import dot, square_root_method


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
        self.mqa = None

    def get_mean_quadratic_approximation(self, n: int = 5, verbose: bool = False):
        if isinstance(self.fsystem, fs.TrigonometricSystem):
            a0, b0 = -pi, pi
            self.approx_function = function_rescale(
                self.approx_function, *self.borders, a0, b0)
        else:
            a0, b0 = self.borders
        matrix, v = self._make_system(a0, b0, n)
        c = square_root_method(matrix, v, verbose=verbose)
        self.mqa = lambda x: np.dot(c, np.array([self.fsystem.get_function(k)(x) for k in range(n)]))
        if verbose:
            self._delta()
        if isinstance(self.fsystem, fs.TrigonometricSystem):
            self.approx_function = function_rescale(self.approx_function, a0, b0, *self.borders)
            self.mqa = function_rescale(self.mqa, a0, b0, *self.borders)
        if verbose:
            self._delta()
        return self.mqa

    def _make_system(self, a0, b0, n):
        v = np.array([dot(a0, b0, self.approx_function, self.fsystem.get_function(i))
                      for i in range(n)])
        matrix = np.array([[dot(a0, b0, self.fsystem.get_function(k), self.fsystem.get_function(j))
                            for j in range(n)] for k in range(n)])
        return matrix, v

    def _delta(self):
        print("||f-Qn||^2 =", util.dot(pi / 2, 3 * pi / 2, lambda x: self.approx_function(x) - self.mqa(x),
                                       lambda x: self.approx_function(x) - self.mqa(x)))

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




