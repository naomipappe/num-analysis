import time
from typing import Callable

import numpy as np


# TODO upper border getter for infinity limit

def upper_border_approximation_ihor(eps: float) -> float:
    return np.ceil(2 / eps)


def upper_border_approximation_sasha(eps: float) -> float:
    arg = (np.pi * np.sqrt(2) - 2 * eps) / (2 * np.sqrt(2))
    return np.ceil((np.tan(arg) - 1) / np.sqrt(2))


def upper_border_approximation_denys(eps: float) -> float:
    return np.ceil(np.tan((np.pi - eps) / 2))


class Integral:
    def __init__(self):
        self._description = "Base integral class"

    @classmethod
    def _algebraic_precision(cls):
        return 0

    def integrate(self, eps: float, verbose: bool, runge: bool):
        pass

    def _apriori_error_measure(self):
        pass

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        return abs(integral_prev - integral_next) / (2 ** self._algebraic_precision() - 1)

    def _runge_method(self, eps: float):
        pass

    def _integrate(self, integrand: Callable[[float], float]):
        pass

    def _richardson_clarification(self, integral_prev, integral_next):
        return 2 ** self._algebraic_precision() / (2 ** self._algebraic_precision() - 1) * integral_next - 1 / (
                2 ** self._algebraic_precision() - 1) * integral_prev


class MeanRectangleIntegral(Integral):
    _algebraic_precision_value = 2

    def __init__(self, a: float, b: float, integrand: Callable[[float], float],
                 integrand_2_derivative: Callable[[float], float]):
        Integral.__init__(self)
        self._description = 'Интеграл функции, вычисленный с помощью составной формулы средних прямоугольников'
        self.borders = a, b
        self.integrand = integrand
        self.integrand_second_derivative = integrand_2_derivative
        self._step = None

    @classmethod
    def _algebraic_precision(cls):
        return cls._algebraic_precision_value

    def integrate(self, eps: float, verbose: bool = False, runge: bool = False):
        if runge:
            if self.borders[1] == np.inf:
                self.borders = self.borders[0], upper_border_approximation_ihor(eps)
            integral, h, aposteriori_error = self._runge_method(eps)
            if verbose:
                print("Априорная оценка погрешности: ", abs(self._apriori_error_measure()))
                print(f"Интеграл от функции по отрезку [{self._a},{self._b}]")
                print(f"I = {integral}, h = {h}")
                print(f"Апостериорная оценка погрешности : ", aposteriori_error)
            return integral
        else:
            if self.borders[1] == np.inf:
                self._b = upper_border_approximation_ihor(eps)
            self._step = (self._b - self._a)
            while self._apriori_error_measure() > eps / 2:
                self._step /= 2
            integral, h = self._integrate(self.integrand)
            if verbose:
                print("Априорная оценка погрешности: ", abs(self._apriori_error_measure()))
                print(f"Интеграл от функции по отрезку [{self._a},{self._b}]")
                print(f"I = {integral}, h = {h}")
            return integral

    def _apriori_error_measure(self):
        d2f_measure = 0
        n = (self.borders[1] - self.borders[0]) / self._step
        for i in range(int(n + 1)):
            x_i = self.borders[0] + i * self._step
            d2f_measure += d2f(x_i - self._step / 2)

        error = (self._step ** 2) / 24 * self._step * d2f_measure
        return abs(error)

    def _integrate(self, integrand: Callable[[float], float]):
        integral = 0
        n = (self.borders[1] - self.borders[0]) / self._step
        for i in range(int(n + 1)):
            x_i = self.borders[0] + i * self._step
            integral += integrand(x_i - self._step / 2)
        return integral * self._step, self._step

    def _runge_method(self, eps: float):
        self._step = (self.borders[1] - self.borders[0])
        integral_prev = self._integrate(self.integrand)
        self._step = self._step / 2
        integral_next = self._integrate(self.integrand)
        while self._aposteriori_error_measure(integral_prev[0], integral_next[0]) > eps / 2:
            self._step = self._step / 2
            integral_prev = integral_next
            integral_next = self._integrate(self.integrand)
        integral_next = super()._richardson_clarification(integral_next=integral_next[0],
                                                          integral_prev=integral_prev[0]), integral_next[1]
        return integral_next[0], integral_next[1], self._aposteriori_error_measure(integral_prev[0], integral_next[0])

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        return super()._aposteriori_error_measure(integral_prev, integral_next)

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
    _algebraic_precision_value = 4

    def __init__(self, a: float, b: float, integrand: Callable[[float], float], d4f: Callable[[float], float]):
        Integral.__init__(self)
        self._description = 'Интеграл функции, вычисленный с помощью составной формулы Симпсона'
        self.borders = a, b
        self.integrand = integrand
        self.integrand_4_derivative = d4f
        self._step = None

    @classmethod
    def _algebraic_precision(cls):
        return cls._algebraic_precision_value

    def integrate(self, eps: float, verbose: bool or None = None, runge: bool = False):
        if runge:
            if self.borders[1] == np.inf:
                self.borders = self.borders[0], upper_border_approximation_ihor(eps)
            integral, h, aposteriori_error = self._runge_method(eps)
            if verbose:
                print("Априорная оценка погрешности: ", abs(self._apriori_error_measure()))
                print(f"Интеграл от функции по отрезку [{self._a},{self._b}]")
                print(f"I = {integral}, h = {h}")
                print(f"Апостериорная оценка погрешности : ", aposteriori_error)
            return integral
        else:
            if self.borders[1] == np.inf:
                self._b = upper_border_approximation_ihor(eps)
            self._step = (self._b - self._a) / 10
            while self._apriori_error_measure() > eps:
                self._step /= 2
            integral, h = self._integrate(self.integrand)
            if verbose:
                print("Априорная оценка погрешности: ", abs(self._apriori_error_measure()))
                print(f"Интеграл от функции по отрезку [{self._a},{self._b}]")
                print(f"I = {integral}, h = {h}")
            return integral

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        return super()._aposteriori_error_measure(integral_prev, integral_next)

    def _apriori_error_measure(self):
        return abs((self._step ** 4 / 2880) * self._integrate(self.integrand_4_derivative)[0])

    def _runge_method(self, eps: float):
        self._step = (self.borders[1] - self.borders[0])
        integral_prev = self._integrate(self.integrand)
        self._step = self._step / 2
        integral_next = self._integrate(self.integrand)
        while self._aposteriori_error_measure(integral_prev[0], integral_next[0]) > eps / 2:
            self._step = self._step / 2
            integral_prev = integral_next
            integral_next = self._integrate(self.integrand)
        integral_next = super()._richardson_clarification(integral_next=integral_next[0],
                                                          integral_prev=integral_prev[0]), integral_next[1]
        return integral_next[0], integral_next[1], self._aposteriori_error_measure(integral_prev[0], integral_next[0])

    def _integrate(self, integrand: Callable[[float], float]):
        integral = 0
        n = int((self.borders[1] - self.borders[0]) / self._step)

        def node(k: int) -> float:
            return self.borders[0] + k * self._step

        for i in range(1, n, 2):
            integral += self.integrand(node(i - 1)) + 4 * self.integrand(node(i)) + self.integrand(node(i + 1))

        return self._step / 3 * integral, self._step

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
    def integrand(self, new_integrand: Callable[[float], float]):
        self._integrand = new_integrand

    @property
    def integrand_4_derivative(self):
        return self._d4f

    @integrand_4_derivative.setter
    def integrand_4_derivative(self, d4f_new: Callable[[float], float]):
        self._d4f = d4f_new


if __name__ == "__main__":
    def f(x):
        return (x ** 2 + 1) / (x ** 4 + 1)


    def g(x):
        return 1 / (x ** 2 + 4 * x + 13)


    def d2f(x):
        return -2 * x * (x ** 4 + 2 * (x ** 2) - 1) / ((x ** 4 + 1) ** 2)


    def d2g(x):
        return -2 * (x + 2) / (g(x) ** 2)


    def d4g(x):
        return 24 * (2 * x + 4) ** 4 / (x ** 2 + 4 * x + 13) ** 5 - 72 * (2 * x + 4) ** 2 / (
                x ** 2 + 4 * x + 13) ** 4 + 24 / (x ** 2 + 4 * x + 13) ** 3


    start = time.time()
    I = SimpsonIntegral(1, np.inf, g, d4g)
    print(I)
    I.integrate(eps=1e-5, verbose=True, runge=True)
    end = time.time()
    print("time:", end - start)
