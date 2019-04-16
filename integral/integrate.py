from typing import Callable

import numpy as np


def upper_border_approximation_ihor(eps: float) -> float:
    return np.ceil(2 / eps)


def upper_border_approximation_sasha(eps: float) -> float:
    arg = (np.pi * np.sqrt(2) - 2 * eps) / (2 * np.sqrt(2))
    return np.ceil((np.tan(arg) - 1) / np.sqrt(2))


def upper_border_approximation_denys(eps: float) -> float:
    return np.ceil(np.tan((np.pi - eps) / 2))


class Integral:
    def __init__(self):
        self._description = 'Base integral class'

    @classmethod
    def _algebraic_precision(cls):
        return 0

    def integrate(self, eps: float, runge: bool, adaptive: bool):
        pass

    def _apriori_error_measure(self, eps):
        pass

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        return abs(integral_prev - integral_next) / (2 ** self._algebraic_precision() - 1)

    def _runge_method(self, eps: float):
        pass

    def _integrate(self, integrand: Callable[[float], float], h: float, borders):
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

    def integrate(self, eps: float, runge: bool = False, adaptive: bool = False):
        if self.borders[1] == np.inf:
            self._b = upper_border_approximation_sasha(eps)
        if adaptive and runge:
            adaptive = False
        if runge:
            print('Метод Рунге')
            if self.borders[1] == np.inf:
                self.borders = self.borders[0], upper_border_approximation_sasha(eps)
            integral, h, aposteriori_error = self._runge_method(eps)
            print(f'Интеграл от функции по отрезку [{self._a},{self._b}]')
            print(f'I = {integral}, h = {h}')
            print(f'Апостериорная оценка погрешности : ', aposteriori_error)
            return integral
        elif adaptive:
            print('Адаптивная квадратурная формула прямоугольников')
            print(f'Интеграл от функции по отрезку [{self._a},{self._b}]')
            integral, steps = self._adaptive_step_integration(eps)
            print(f'I = {integral}')
            print('Адаптивные шаги интегрирования:')
            print(" ; ".join(map(str, steps)))
            return integral
        else:
            self._step = (self.borders[1] - self.borders[0]) / 10
            error = self._apriori_error_measure(eps)
            while error > eps:
                self._step /= 2
                error = self._apriori_error_measure(eps)
            integral, h = self._integrate(self.integrand, self._step, self.borders)
            print('Априорная оценка погрешности: ', error)
            print(f'Интеграл от функции по отрезку [{self._a},{self._b}]')
            print(f'I = {integral}, h = {h}')
            return integral

    def _apriori_error_measure(self, eps):
        vals = np.linspace(self.borders[0], self.borders[1], min(int(1 / eps), 10 ** 5))
        vals = np.vectorize(self._d2f)(vals)
        d2f_measure = max(abs(vals))
        error = (self._step ** 2) * (self.borders[1] - self.borders[0]) * d2f_measure / 24
        return abs(error)

    def _integrate(self, integrand: Callable[[float], float], h: float, borders):
        integral = 0
        n = np.ceil((borders[1] - borders[0]) / h)
        for i in range(int(n)):
            x_i = borders[0] + i * h
            integral += integrand(x_i - h / 2)
        return integral * h, h

    def _runge_method(self, eps: float):
        self._step = (self.borders[1] - self.borders[0])
        integral_prev = self._integrate(self.integrand, self._step, self.borders)
        self._step = self._step / 2
        integral_next = self._integrate(self.integrand, self._step, self.borders)
        while self._aposteriori_error_measure(integral_prev[0], integral_next[0]) > eps / 2:
            self._step = self._step / 2
            integral_prev = integral_next
            integral_next = self._integrate(self.integrand, self._step, self.borders)
        integral_next = super()._richardson_clarification(integral_next=integral_next[0],
                                                          integral_prev=integral_prev[0]), integral_next[1]
        return integral_next[0], integral_next[1], self._aposteriori_error_measure(integral_prev[0], integral_next[0])

    def _adaptive_step_integration(self, eps: float):
        integral = 0
        left_border = self.borders[0]
        h = (self.borders[1] - self.borders[0]) / 10
        adaptive_steps = []
        while True:
            while True:
                rect = (left_border + h - left_border) * self.integrand((left_border + left_border + h) / 2)
                rect_comp = (left_border + h / 2 - left_border) * self.integrand(
                    (left_border + left_border + h / 2) / 2) + (
                                    left_border + h - left_border - h / 2) * self.integrand(
                    (left_border + h / 2 + left_border + h) / 2)

                if self._aposteriori_error_measure(rect, rect_comp) <= h * eps / (self.borders[1] - self.borders[0]):
                    break

                else:
                    h /= 2
            adaptive_steps.append(h)
            left_border += h
            integral += rect_comp
            h *= 2
            if left_border == self.borders[1]:
                break
            if (left_border + h) > self.borders[1]:
                h = self.borders[1] - left_border
        return integral, adaptive_steps

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        return super()._aposteriori_error_measure(integral_prev, integral_next)

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

    def integrate(self, eps: float, runge: bool = False, adaptive: bool = False):
        if self.borders[1] == np.inf:
            self._b = upper_border_approximation_ihor(eps)
        if runge and adaptive:
            adaptive = False
            print("Runge")
        if runge:
            integral, h, aposteriori_error = self._runge_method(eps)
            print('Априорная оценка погрешности: ', abs(self._apriori_error_measure()))
            print(f'Интеграл от функции по отрезку [{self._a},{self._b}]')
            print(f'I = {integral}, h = {h}')
            print(f'Апостериорная оценка погрешности : ', aposteriori_error)
            return integral
        elif adaptive:
            print('Адаптивная формула Симпсона')
            print(f'Интеграл от функции по отрезку [{self._a},{self._b}]')
            integral, steps = self._adaptive_step_integration(eps)
            print(f'I = {integral}')
            print('Адаптивные шаги интегрирования:')
            print(" ; ".join(map(str, steps)))
            return integral
        else:
            self._step = (self._b - self._a) / 10
            while self._apriori_error_measure() > eps:
                self._step /= 2
            integral, h = self._integrate(self.integrand, self._step, self.borders)
            print('Априорная оценка погрешности: ', abs(self._apriori_error_measure()))
            print(f'Интеграл от функции по отрезку [{self._a},{self._b}]')
            print(f'I = {integral}, h = {h}')
            return integral

    def _aposteriori_error_measure(self, integral_prev: float, integral_next: float):
        return super()._aposteriori_error_measure(integral_prev, integral_next)

    def _apriori_error_measure(self):
        return abs((self._step ** 4 / 2880) * self._integrate(self.integrand_4_derivative, self._step, self.borders)[0])

    def _runge_method(self, eps: float):
        self._step = (self.borders[1] - self.borders[0])
        integral_prev = self._integrate(self.integrand, self._step, self.borders)
        self._step = self._step / 2
        integral_next = self._integrate(self.integrand, self._step, self.borders)
        while self._aposteriori_error_measure(integral_prev[0], integral_next[0]) > eps / 2:
            self._step = self._step / 2
            integral_prev = integral_next
            integral_next = self._integrate(self.integrand, self._step, self.borders)
        integral_next = super()._richardson_clarification(integral_next=integral_next[0],
                                                          integral_prev=integral_prev[0]), integral_next[1]
        return integral_next[0], integral_next[1], self._aposteriori_error_measure(integral_prev[0], integral_next[0])

    def _adaptive_step_integration(self, eps: float):
        integral = 0
        border_left = self.borders[0]
        h = (self.borders[1] - self.borders[0]) / 10
        adaptive_steps = []
        while True:
            while True:
                i_prev = self._simspon_simple(border_left, border_left + h)
                i_next = self._simspon_simple(border_left, border_left + h / 2) + self._simspon_simple(
                    border_left + h / 2, border_left + h)
                if self._aposteriori_error_measure(i_prev, i_next) < h * eps / (self.borders[1] - self.borders[0]):
                    break
                else:
                    h /= 2
            adaptive_steps.append(h)
            border_left += h
            integral += i_next
            h *= 2
            if border_left == self.borders[1]:
                break
            if (border_left + h) > self.borders[1]:
                h = self.borders[1] - border_left
        return integral, adaptive_steps

    def _integrate(self, integrand: Callable[[float], float], h: float, borders):
        integral = 0
        n = int((borders[1] - borders[0]) / h)

        def node(k: int) -> float:
            return borders[0] + k * h

        for i in range(0, n, 2):
            integral += self.integrand(node(i)) + 4 * self.integrand(node(i + 1)) + self.integrand(node(i + 2))

        return h / 3 * integral, h

    def _simspon_simple(self, border_left, border_right):
        return (border_right - border_left) / 6 * (
                self.integrand(border_left) + 4 * self.integrand((border_left + border_right) / 2) + self.integrand(
            border_right))

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


    t = SimpsonIntegral(1, np.inf, g, d4g)
    print(t.integrate(1e-5, False, True))
    print("Истинное :", np.pi / 12)
