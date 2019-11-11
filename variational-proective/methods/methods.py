from integral.integrate import Integral
from integral.integration_strategy import IntegrationStrategy
from integral.integration_formulas import QuadraticFormula
from typing import Type, Tuple, Callable
from sympy import lambdify
from functional.fun_sys import FunctionalSystem
import numpy as np
from scipy import integrate


class VariationalProective:
    formula: Type[QuadraticFormula]
    strategy: Type[IntegrationStrategy]
    context: dict
    constants: dict
    fsys: Type[FunctionalSystem]
    integral = Integral()

    @classmethod
    def solve(
        cls, context: dict, n: int, tolerance=1e-3
    ) -> Tuple[Callable[[float], float], list]:
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
    def solve(cls, context: dict, n: int, tolerance: float or None):
        cls.context = context
        if tolerance is None:
            tolerance = 1e-3
        cls.integral.borders = context["borders"]
        matrix, vector, phi_funcs = cls.__build_system(n, tolerance)
        coeficients = np.linalg.solve(matrix, vector)
        error = vector - np.dot(matrix, coeficients)
        print(np.linalg.norm(error))
        def approximation(x: float) -> float:
            return lambdify(
                cls.context["variable"], cls.fsys.get_basic_zero(), "numpy"
            )(x) + sum([coeficients[i] * phi_funcs[i](x) for i in range(n)])

        return approximation

    @classmethod
    def __build_system(cls, n: int, tolerance):
        modified_f_expr = lambdify(
            cls.context["variable"],
            cls.context["L"](
                cls.context["solution_exact_expr"], cls.context["variable"]
            ).simplify()
            - cls.context["L"](
                cls.fsys.get_basic_zero(), cls.context["variable"]
            ).simplify(),
            "numpy",
        )
        phi_funcs = [
            lambdify(cls.context["variable"], cls.fsys.get_function(i), "numpy")
            for i in range(n)
        ]
        L_phi_funcs = [
            lambdify(
                cls.context["variable"],
                cls.context["L"](cls.fsys.get_function(i), cls.context["variable"]),
                "numpy",
            )
            for i in range(n)
        ]
        matrix = np.matrix(
            [
                [
                    cls.__integration(L_phi_funcs[j], phi_funcs[i], tolerance)
                    for j in range(n)
                ]
                for i in range(n)
            ]
        )
        vector = [
            cls.__integration(modified_f_expr, phi_funcs[i], tolerance) for i in range(n)
        ]
        return matrix, vector, phi_funcs

    @classmethod
    def __integration(
        cls, f: Callable[[float], float], g: Callable[[float], float], tolerance: float
    ) -> float:
        cls.integral.integrand = lambda x: f(x) * g(x)
        return cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        #return integrate.quad(cls.integral.integrand,*cls.integral.borders)[0]


class Collocation(VariationalProective):
    pass


class LeastSquares(VariationalProective):
    @classmethod
    def solve(cls):
        raise NotImplementedError("Least Squares")


class BubnovGalerkin(VariationalProective):
    @classmethod
    def solve(cls):
        raise NotImplementedError("BubnovGalerkin")

