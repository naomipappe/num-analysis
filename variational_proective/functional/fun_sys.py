from typing import Tuple

class FunctionalSystem:
    def __init__(self, borders: tuple, alpha: float, beta: float, gamma: float, delta: float, k, mu_1, mu_2, variable):
        self._variable = variable
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._k = k
        self._mu_1 = mu_1
        self._mu_2 = mu_2
        self._border_left, self._border_right = borders
        self._C = self._border_right + self._k(self._border_right) * (
                self._border_right - self._border_left) / (2 * self._k(self._border_right)
                                                           + delta * (self._border_right - self._border_left))
        self._D = self._border_left - self._k(self._border_left) * (self._border_right - self._border_left) \
                  / (2 * self._k(self._border_left) + beta * (self._border_right - self._border_left))

        # Check whether we have homogeneous case
        if self._mu_1() == 0 and self._mu_2() == 0:
            self._A_psi = 0
            self._B_psi = 0
        else:
            self._A_psi = (self._mu_1() - self._mu_2() * beta / delta) / (
                    -self._k(self._border_left) + beta * self._border_left - (beta / delta)
                    * (self._k(self._border_right) + delta * self._border_right))
            self._B_psi = (self._mu_2() - self._A_psi * (
                    self._k(self._border_right) + delta * self._border_right)) / delta

    def get_function(self, j: int):
        raise NotImplementedError

    def get_basic_zero(self):
        return self._A_psi * self._variable + self._B_psi

    def get_derivative(self, j: int, order: int):
        return self.get_function(j).diff(self._variable, order)

    @property
    def variable(self):
        return self._variable

    @property
    def borders(self) -> Tuple:
        return self._border_left, self._border_right


class BasisFunction(FunctionalSystem):
    def __init__(self, borders: tuple, alpha: float, beta: float, gamma: float, delta: float, k, mu_1, mu_2, variable):
        FunctionalSystem.__init__(self, borders, alpha, beta, gamma, delta, k, mu_1, mu_2, variable)

    def get_function(self, j):
        if j == 0:
            return ((self.variable - self.borders[0]) ** 2) * (self.variable - self._C)
        elif j == 1:
            return ((self.borders[1] - self.variable) ** 2) * (self.variable - self._D)
        else:
            return ((self.variable - self.borders[0]) ** j) * ((self.borders[1] - self.variable) ** 2)


class EvenOddBasis(FunctionalSystem):
    def __init__(self, borders: tuple, alpha: float, beta: float, gamma: float, delta: float, k, mu_1, mu_2, variable):
        super().__init__(borders, alpha, beta, gamma, delta, k, mu_1, mu_2, variable)

    def get_function(self, j):
        if j % 2 == 0:
            B = self._border_left - self._alpha * (self._border_right - self._border_left) / (
                    self._alpha * j + self._beta * (self._border_right - self._border_left))
            return (self._variable - B) * (self._border_right - self._variable) ** j
        else:
            A = self._gamma * (self._border_right - self._border_left) / (
                    self._gamma * (j + 1) + self._delta * (self._border_right - self._border_left))
            return (self._variable - A) * (self._variable - self._border_left) ** (j + 1)


class AnotherSystem(FunctionalSystem):
    def __init__(self, borders: tuple, alpha: float, beta: float, gamma: float, delta: float, k, mu_1, mu_2, variable):
        super().__init__(borders, alpha, beta, gamma, delta, k, mu_1, mu_2, variable)

    def get_function(self, j: int):
        if j == 0:
            return ((self._variable - self._border_left) ** 2) * (self._variable - self._C)
        if j == 1:
            return ((self._border_right - self._variable) ** 2) * (self._variable - self._D)
        return (self._variable - self._border_left) ** 2 * ((self._border_right - self._variable) ** j)
