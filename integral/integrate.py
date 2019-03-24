from typing import Callable

import numpy as np


class Integral:
    def __init__(self):
        self._description = "Base integral class"

    def integrate(self, n: int):
        pass

    def _apriori_error_measure(self):
        pass

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        pass

    def _runge_method(self, eps: float):
        pass


# TODO change method of calculation integral value, due to bad performance on 1e-3 accuracy already

class MeanRectangleIntegral(Integral):
    def __init__(self, a: float, b: float, f: Callable[[float], float], d2f: Callable[[float], float]):
        Integral.__init__(self)
        self._description = "Integral of function, calculated by mean rectangle formula"
        self.borders = a, b
        self.integrand = f
        self.integrand_second_derivative = d2f
        self._nodes = None
        self._step = None

    def integrate(self, eps: float or None = None, verbose: bool = False):
        if eps is not None:
            if self.borders[1] == np.inf:
                self.borders = self.borders[0], 2 / eps
            integral = self._runge_method(eps)
            if verbose:
                print("Априорная оценка погрешности: ", abs(self._apriori_error_measure()))
                print(f"Интеграл от функции по отрезку [{self._a},{self._b}]")
                print(f"I = {integral[0]}, h = {integral[1]}")
                print(f"Апостериорная оценка погрешности : ", integral[2])
            return integral[0]
        else:
            self._recalculate_nodes(200)
            integral = self._integrate()
            if verbose:
                print("Априорная оценка погрешности: ", abs(self._apriori_error_measure()))
                print(f"Интеграл от функции по отрезку [{self._a},{self._b}]")
                print(f"I = {integral[0]}, h = {integral[1]}")
            return integral[0]

    def _apriori_error_measure(self):
        error = self._step ** 2 / 24 * self._step * sum(map(d2f, self._nodes))
        return error

    def _recalculate_nodes(self, n: int):
        self._nodes, self._step = np.linspace(*self.borders, n, retstep=True)
        self._nodes = [x - self._step / 2 for x in self._nodes]

    def _integrate(self):
        return self._step * sum(map(self.integrand, self._nodes)), self._step

    def _runge_method(self, eps: float):
        n = 2
        self._recalculate_nodes(n)
        integral_prev = self._integrate()
        self._recalculate_nodes(2 * n)
        integral_next = self._integrate()
        while self._aposteriori_error_measure(integral_prev[0], integral_next[0]) >= eps:
            n = 2 * n
            integral_prev = integral_next
            self._recalculate_nodes(2 * n)
            integral_next = self._integrate()
        return integral_next[0], integral_next[1], self._aposteriori_error_measure(integral_prev[0], integral_next[0])

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        return abs(integral_next - integral_prev)

    @property
    def description(self):
        return self._description

    @property
    def borders(self):
        return self._a, self._b

    @borders.setter
    def borders(self, borders: tuple):
        if borders[0] > borders[1]:
            self._a, self._b = reversed(borders)
        else:
            self._a, self._b = borders

    @property
    def integrand(self):
        return self._integrand

    @integrand.setter
    def integrand(self, f_new: Callable[[float], float]):
        self._integrand = f_new

    @property
    def integrand_second_derivative(self):
        return self._d2f

    @integrand_second_derivative.setter
    def integrand_second_derivative(self, d2f_new: Callable[[float], float]):
        self._d2f = d2f_new


class SimpsonIntegral(Integral):
    def __init__(self, a: float, b: float, f: Callable[[float], float], d2f: Callable[[float], float]):
        Integral.__init__(self)
        self._description = "Integral of function calculated using Simpson's rule"
        self.borders = a, b
        self.integrand = f
        self.d2f = d2f
        self._nodes = None
        self._step = None

    def integrate(self, n: int):
        pass

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        pass

    def _apriori_error_measure(self):
        pass

    def _runge_method(self, eps: float):
        pass

    def _integrate(self):
        pass

    def __str__(self):
        return self.description

    @property
    def description(self):
        return self._description

    @property
    def borders(self):
        return self._a, self._b

    @borders.setter
    def borders(self, borders: tuple):
        if borders[0] > borders[1]:
            self._a, self._b = reversed(borders)
        else:
            self._a, self._b = borders

    @property
    def integrand(self):
        return self._integrand

    @integrand.setter
    def integrand(self, f_new: Callable[[float], float]):
        self._integrand = f_new

    @property
    def d2f(self):
        return self._d2f

    @d2f.setter
    def d2f(self, d2f_new: Callable[[float], float]):
        self._d2f = d2f_new


if __name__ == "__main__":
    def f(x):
        return (x ** 2 + 1) / (x ** 4 + 1)


    def d2f(x):
        return -2 * x * (x ** 4 + 2 * (x ** 2) - 1) / (x ** 4 + 1)


    I = MeanRectangleIntegral(0, np.inf, f, d2f)
    print(I.integrate(eps=1e-3, verbose=True))
