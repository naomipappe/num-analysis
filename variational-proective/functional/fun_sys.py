from typing import Callable
from sympy import lambdify


class FunctionalSystem:
    def __init__(self, context: dict):
        super().__init__()
        self.context = context

    def get_function(self, k: int):
        raise NotImplementedError

    def get_basic_zero(self):
        raise NotImplementedError

    def get_derrivative(self, k: int, order: int):
        raise NotImplementedError


class BasisFunction(FunctionalSystem):
    def __init__(self, context):
        super().__init__(context)
        self.__constants = context["constants"]
        self.__context = context
        self.__x = context["variable"]
        self.__a, self.__b = context["borders"]
        self.__C = self.__b + (
            context["k(x)"](self.__b)
            * (self.__b - self.__a)
            / (
                2 * context["k(x)"](self.__b)
                + self.__constants["delta"] * (self.__b - self.__a)
            )
        )
        self.__D = self.__a - (
            context["k(x)"](self.__a)
            * (self.__b - self.__a)
            / (
                2 * context["k(x)"](self.__a)
                + self.__constants["beta"] * (self.__b - self.__a)
            )
        )
        # check if we have a homogeneous
        if context["mu_1"]() == 0 and context["mu_2"]() == 0:
            self.__A_psi = 0
            self.__B_psi = 0
        else:
            self.__A_psi = (
                context["mu_1"]()
                - context["mu_2"]()
                * self.__constants["beta"]
                / self.__constants["delta"]
            ) / (
                -self.context["k(x)"](self.__a)
                + self.__constants["beta"] * self.__a
                - (self.__constants["beta"] / self.__constants["delta"])
                * (
                    self.context["k(x)"](self.__b)
                    + self.__constants["delta"] * self.__b
                )
            )
            self.__B_psi = (
                self.context["mu_2"]()
                - self.__A_psi
                * (
                    self.context["k(x)"](self.__b)
                    + self.__constants["delta"] * self.__b
                )
            ) / self.__constants["delta"]

    def get_basic_zero(self):
        return self.__A_psi * self.__x + self.__B_psi

    def get_function(self, k):
        if k == 0:
            return ((self.__x - self.__a) ** 2) * (self.__x - self.__C)
        elif k == 1:
            return ((self.__b - self.__x) ** 2) * (self.__x - self.__D)
        else:
            return ((self.__x - self.__a) ** (k - 1)) * ((self.__b - self.__x) ** 2)

    def get_derrivative(self, k, order):
        return self.get_function(k).diff(self.__x, order)


class TestFunction(FunctionalSystem):
    def __init__(self, context):
        super().__init__(context)
        self.__constants = context["constants"]
        self.__context = context
        self.__x = context["variable"]
        self.__a, self.__b = context["borders"]
        # check if we have a homogeneous
        if context["mu_1"]() == 0 and context["mu_2"]() == 0:
            self.__A_psi = 0
            self.__B_psi = 0
        else:
            self.__A_psi = (
                context["mu_1"]()
                - context["mu_2"]()
                * self.__constants["beta"]
                / self.__constants["delta"]
            ) / (
                -self.context["k(x)"](self.__a)
                + self.__constants["beta"] * self.__a
                - (self.__constants["beta"] / self.__constants["delta"])
                * (
                    self.context["k(x)"](self.__b)
                    + self.__constants["delta"] * self.__b
                )
            )
            self.__B_psi = (
                self.context["mu_2"]()
                - self.__A_psi
                * (
                    self.context["k(x)"](self.__b)
                    + self.__constants["delta"] * self.__b
                )
            ) / self.__constants["delta"]

    def get_basic_zero(self):
        return self.__A_psi * self.__x + self.__B_psi

    def get_function(self, k):
        if k % 2 == 0:
            B = self.__a - self.__constants["alpha"] * (self.__b - self.__a) / (
                self.__constants["alpha"] * k
                + self.__constants["beta"] * (self.__b - self.__a)
            )
            return (self.__x - B) * (self.__b - self.__x) ** k
        else:
            A = (
                self.__constants["gamma"]
                * (self.__b - self.__a)
                / (
                    self.__constants["gamma"] * (k + 1)
                    + self.__constants["delta"] * (self.__b - self.__a)
                )
            )
            return (self.__x - A) * (self.__x - self.__a) ** (k + 1)

    def get_derrivative(self, k, order):
        return self.get_function(k).diff(self.__x, order)
