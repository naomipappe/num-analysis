from integral.integrate import Integral
from integral.integration_strategy import IntegrationStrategy
from integral.integration_formulas import QuadraticFormula
from typing import Type
from typing import Callable
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
    def solve(cls, context: dict, n: int):
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
    def solve(cls, context: dict, n: int):
        cls.integral.borders = context["borders"]
        cls.context = context
        cls.constants = context["constants"]

        E = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                E[i][j] = cls.__G_function(i, j)

        print(E)

    @classmethod
    def __G_function(cls, i: int, j: int):
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
        result = cls.integral.integrate(1e-3, cls.strategy, cls.formula)
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
    def __L_function(cls, v: Callable[[float], float]):
        def l(x: float) -> float:
            return cls.context["new_Ritz_L"](x) * v(x)

        cls.integral.integrand = l
        result = cls.integral.integrate(1e-3, cls.strategy, cls.formula)
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

