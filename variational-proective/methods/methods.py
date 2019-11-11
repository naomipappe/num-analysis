from integral.integrate import Integral
from integral.integration_strategy import IntegrationStrategy
from integral.integration_formulas import QuadraticFormula
from typing import Type, Tuple, Callable
from sympy import lambdify
from functional.fun_sys import FunctionalSystem
import numpy as np


class VariationalProective:
    formula: Type[QuadraticFormula]
    strategy: Type[IntegrationStrategy]
    context: dict
    constants: dict
    fsys: Type[FunctionalSystem]
    integral = Integral()

    @classmethod
    def solve(cls, context: dict, n: int) -> Tuple[Callable[[float], float], list]:
        raise NotImplementedError

    @classmethod
    def set_integration_method(
        cls, formula: Type[QuadraticFormula], strategy: Type[IntegrationStrategy]
    ):
        cls.formula = formula
        cls.strategy = strategy

    @classmethod
    def set_functional_system(cls, fsys: Type[FunctionalSystem]):
        cls.fsys = fsys


class Ritz(VariationalProective):
    @classmethod
    def solve(
        cls, context: dict, n: int, tolerance: float = 1e-3
    ) -> Tuple[Callable[[float], float], list]:
        cls.integral.borders = context["borders"]
        cls.context = context
        cls.constants = context["constants"]

        E = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                E[i][j] = cls.__G_function(i + 1, j + 1, tolerance)

        F = np.zeros((n, 1))
        for i in range(0, n):
            F[i] = cls.__L_function(cls.fsys.get_function(i + 1))

        coeficients = np.linalg.solve(E, F)
        error = F - np.dot(E, coeficients)

        def approximation(x: float) -> float:
            return cls.fsys.basic_zero(x) + sum(
                [
                    coeficients[i] * cls.fsys.get_function(i + 1)(x)
                    for i in range(len(coeficients))
                ]
            )

        return approximation, error

    @classmethod
    def __G_function(cls, i: int, j: int, tolerance: float or None = None) -> float:
        if tolerance == None:
            tolerance = 1e-3

        def g(x: float):
            return (
                lambdify(cls.context["variable"], cls.context["k(x)"](), "numpy")(x)
                * cls.fsys.get_first_derivative_function(i)(x)
                * cls.fsys.get_first_derivative_function(j)(x)
            )
            +cls.context["q(x)"](x) * cls.fsys.get_function(i)(
                x
            ) * cls.fsys.get_function(j)(x)

        cls.integral.integrand = g
        result = cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        result += (
            cls.constants["alpha_1"]
            * cls.fsys.get_function(i)(cls.integral.borders[0])
            * cls.fsys.get_function(j)(cls.integral.borders[0])
        )
        result += (
            cls.constants["alpha_2"]
            * cls.fsys.get_function(i)(cls.integral.borders[1])
            * cls.fsys.get_function(j)(cls.integral.borders[1])
        )
        return result

    @classmethod
    def __L_function(
        cls, v: Callable[[float], float], tolerance: float or None = None
    ) -> float:
        if tolerance == None:
            tolerance = 1e-3

        def l(x: float) -> float:
            return cls.context["new_Ritz_L"](x) * v(x)

        cls.integral.integrand = l
        result = cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        return result


class Collocation(VariationalProective):
    @classmethod
    def solve(cls):
        raise NotImplementedError("Collocation")


class LeastSquares(VariationalProective):
    @classmethod
    def solve(cls):
        raise NotImplementedError("Least Squares")


class BubnovGalerkin(VariationalProective):
    @classmethod
    def solve(cls):
        raise NotImplementedError("BubnovGalerkin")

