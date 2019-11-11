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

    def get_first_derivative_function(self, k: int):
        raise NotImplementedError

    def get_second_derivative_function(self, k: int):
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
                + self.__constants["alpha_2"] * (self.__b - self.__a)
            )
        )
        self.__D = self.__a - (
            context["k(x)"](self.__a)
            * (self.__b - self.__a)
            / (
                2 * context["k(x)"](self.__a)
                + self.__constants["alpha_1"] * (self.__b - self.__a)
            )
        )
        self.__A_psi = (
            context["mu_1"]()
            - context["mu_2"]()
            * self.__constants["alpha_1"]
            / self.__constants["alpha_2"]
        ) / (
            -self.context["k(x)"](self.__a)
            + self.__constants["alpha_1"] * self.__a
            - (self.__constants["alpha_1"] / self.__constants["alpha_2"])
            * (self.context["k(x)"](self.__b) + self.__constants["alpha_2"] * self.__b)
        )
        self.__B_psi = (
            self.context["mu_2"]()
            - self.__A_psi
            * (self.context["k(x)"](self.__b) + self.__constants["alpha_2"] * self.__b)
        ) / self.__constants["alpha_2"]

    def get_basic_zero(self):
        return self.__A_psi * self.__x + self.__B_psi

    def get_function(self, k):
        if k == 0:
            return ((self.__x - self.__a) ** 2) * (self.__x - self.__C)
        elif k == 1:
            return ((self.__b - self.__x) ** 2) * (self.__x - self.__D)
        else:
            return ((self.__x - self.__a) ** (k - 1)) * ((self.__b - self.__x) ** 2)

