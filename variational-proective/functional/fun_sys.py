from typing import Callable
from sympy import lambdify


class FunctionalSystem:
    def __init__(self, context: dict):
        super().__init__()
        self.context = context

    def get_function(self, k: int):
        raise NotImplementedError


class BasisFunction(FunctionalSystem):
    def __init__(self, context: dict):
        super().__init__(context)
        self.__constants = self.context["constants"]
        self.__k = lambdify("x", self.context["k(x)"](), "numpy")
        self.__borders = self.context["borders"]

        self.__C = self.__borders[1] + self.__k(self.__borders[1]) * (
            self.__borders[1] - self.__borders[0]
        ) / (
            2 * self.__k(self.__borders[1])
            + self.__constants["alpha_2"] * (self.__borders[1] - self.__borders[0])
        )

        self.__D = self.__borders[0] - self.__k(self.__borders[0]) * (
            self.__borders[1] - self.__borders[0]
        ) / (
            2 * self.__k(self.__borders[0])
            + self.__constants["alpha_1"] * (self.__borders[1] - self.__borders[0])
        )
        # check if we got homogoneous case
        if self.context["mu_1"] == 0 and self.context["mu_2"] == 0:

            def A_psi_homogoneous():
                return 0

            def B_psi_homogoneous():
                return 0

            self.__A_psi = A_psi_homogoneous
            self.__B_psi = B_psi_homogoneous

    def get_function(self, k: int) -> Callable[[int], float]:
        if k == 1:
            return lambda x: (x - self.__C)(x - self.__borders[0]) ** 2
        elif k == 2:
            return lambda x: (x - self.__D) * (self.__borders[1] - x) ** 2
        else:
            return lambda x: ((self.__borders[1] - x) ** 2) * (
                (x - self.__borders[0]) ** (k - 1)
            )

    def basic_zero(self, x: float) -> float:
        return self._A_psi() * x + self._B_psi()


    def __A_psi(self) -> float:
        return (
            self.context["mu_1"]()
            - self.__constants["alpha_1"]
            * self.context["mu_2"]()
            / self.__constants["alpha_2"]
        ) / (
            -self.__k(self.__borders[0])
            + self.__constants["alpha_1"] * self.__borders[0]
            - (self.__constants["alpha_1"] / self.__constants["alpha_2"])
            * (
                self.__k(self.__borders[1])
                + self.__constants["alpha_2"] * self.__borders[1]
            )
        )

    def __B_psi(self) -> float:
        return (
            self.context["mu_2"]()
            - self._A_psi()
            * (
                self.__k(self.__borders[1])
                + self.__constants["alpha_2"] * self.__borders[1]
            )
        ) / self.__constants["alpha_2"]

    def __d_basic_zero(self, x: float) -> float:
        return self._A_psi()

    def __d2_basic_zero(self, x: float) -> float:
        return 0

    def __d_basic_first(self, x: float) -> float:
        return 2 * (x - self.__borders[0]) * (x - self.__C) + (x - self.__borders[0]) ** 2

    def __d2_basic_first(self, x: float) -> float:
        return 2 * (x - self.__C) + 4 * (x - self.__borders[0])

    def __d_basic_second(self, x: float) -> float:
        return -2 * (self.__borders[1] - x) * (x - self.__D) + (self.__borders[1] - x) ** 2

    def __d2_basic_second(self, x: float) -> float:
        return 2 * (x - self.__D) - 4(self.__borders[1] - x)
