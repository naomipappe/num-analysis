from typing import Callable
from funsys import function_rescale
import numpy as np
from util import dot

class LegandreApproximation:
    def __init__(self, a: float, b: float, f: Callable[[float], float]):
        self._a = a
        self._b = b
        self._approx = f
        self._legandre = None

    def get_legandre_approximation_continuous(self, n: int = 5, verbose: bool = False):
        a0 = -1
        b0 = 1
        self._approx = function_rescale(self._approx, self._a, self._b, a0, b0)
        c = np.array([dot(a0, b0, self._approx, self._get_legandre_polynom(i))*(2 * i + 1) / 2 for i in range(n+1)])

        self._legandre = lambda x: np.dot(c, np.array([self._get_legandre_polynom(k)(x) for k in range(n+1)]))

        if verbose:
            self._delta()

        self._approx = function_rescale(self._approx, a0, b0, self._a, self._b)
        self._legandre = function_rescale(self._legandre, a0, b0, self._a, self._b)

        if verbose:
            self._delta()

        return self._legandre

    def _get_legandre_polynom(self, k: int):
        if k == 0:
            return lambda x: 1
        if k == 1:
            return lambda x: x

        return lambda x: (2 * k - 1) / k * x * self._get_legandre_polynom(k - 1)(x) - \
                                      (k - 1) / k * self._get_legandre_polynom(k - 2)(x)

    def _delta(self):
        print("||f-Qn||^2 =", dot(self._a, self._b, lambda x: self._approx(x) - self._legandre(x),
                                  lambda x: self._approx(x) - self._legandre(x)))
