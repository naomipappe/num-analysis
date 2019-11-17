from integral.integrate import Integral
from integral.integration_strategy import IntegrationStrategy
from integral.integration_formulas import QuadraticFormula
from typing import Type, Tuple, Callable
from sympy import lambdify
from functional.fun_sys import FunctionalSystem
import numpy as np
from scipy import integrate

# TODO speed up Ritz
# TODO DRY principle


class VariationalProective:
    formula: Type[QuadraticFormula] = None
    strategy: Type[IntegrationStrategy] = None
    context: dict = None
    constants: dict = None
    fsys: Type[FunctionalSystem] = None
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
        cls.constants = context["constants"]
        if tolerance is None:
            tolerance = 1e-3
        cls.integral.borders = context["borders"]
        matrix, vector = cls.__build_system(n, tolerance)
        coeficients = np.linalg.solve(matrix, vector)
        error = vector - np.dot(matrix, coeficients)

        def approximation(x: float) -> float:
            return lambdify(  # TODO DRY principle
                cls.context["variable"],
                cls.fsys.get_basic_zero()
                + sum([coeficients[i] * cls.fsys.get_function(i) for i in range(n)]),
                "numpy",
            )(x)

        return approximation, error

    @classmethod
    def __build_system(cls, n: int, tolerance):
        L_operator = cls.context["L"]
        variable = cls.context["variable"]

        def f_modified():
            return L_operator(
                cls.context["solution_exact_expr"], variable
            ) - L_operator(cls.fsys.get_basic_zero(), variable)

        matrix = np.matrix(
            [
                [
                    cls.__integration(
                        lambdify(
                            variable,
                            L_operator(cls.fsys.get_function(i), variable)
                            * cls.fsys.get_function(j),
                            "numpy",
                        ),
                        tolerance,
                    )
                    for i in range(n)
                ]
                for j in range(n)
            ]
        )
        vector = np.array(
            [
                cls.__integration(
                    lambdify(variable, f_modified() * cls.fsys.get_function(i), "numpy")
                )
                for i in range(n)
            ]
        )
        return matrix, vector

    @classmethod
    def __integration(
        cls, f: Callable[[float], float], tolerance: float = 1e-3
    ) -> float:
        cls.integral.integrand = f
        if cls.strategy is not None and cls.formula is not None:
            return cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        return integrate.quad(cls.integral.integrand, *cls.integral.borders)[0]


class Collocation(VariationalProective):
    __nodes = None

    @classmethod
    def solve(cls, context, n, tolerance=1e-3):
        cls.context = context
        matrix, vector = cls.__build_system(n)
        coeficients = np.linalg.solve(matrix, vector)
        error = vector - np.dot(matrix, coeficients)

        def approximation(x: float) -> float:
            return lambdify(
                cls.context["variable"],
                cls.fsys.get_basic_zero()
                + sum([coeficients[i] * cls.fsys.get_function(i) for i in range(n)]),
                "numpy",
            )(x)

        return approximation, error

    @classmethod
    def __build_system(cls, n: int):
        def f_modified(x: float) -> float:
            return lambdify(
                cls.context["variable"],
                cls.context["L"](
                    cls.context["solution_exact_expr"], cls.context["variable"]
                )
                - cls.context["L"](cls.fsys.get_basic_zero(), cls.context["variable"]),
                "numpy",
            )(x)

        if cls.__nodes is None:
            cls.__nodes = np.linspace(*cls.context["borders"], n, endpoint=True)
        else:
            n = len(cls.__nodes)

        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = lambdify(
                    cls.context["variable"],
                    cls.context["L"](cls.fsys.get_function(j), cls.context["variable"]),
                    "numpy",
                )(cls.__nodes[i])
        vector = np.array([f_modified(cls.__nodes[i]) for i in range(n)])
        return matrix, vector


class LeastSquares(VariationalProective):
    @classmethod
    def solve(cls, context: dict, n: int, tolerance: float = 1e-3):
        cls.context = context
        cls.constants = context["constants"]
        cls.integral.borders = context["borders"]
        matrix, vector = cls.__build_system(n, tolerance)
        coeficients = np.linalg.solve(matrix, vector)
        error = vector - np.dot(matrix, coeficients)

        def approximation(x: float) -> float:
            return lambdify(
                cls.context["variable"],
                cls.fsys.get_basic_zero()
                + sum([coeficients[i] * cls.fsys.get_function(i) for i in range(n)]),
                "numpy",
            )(x)

        return approximation, error

    @classmethod
    def __build_system(cls, n, tolerance=1e-3):
        L_operator = cls.context["L"]
        variable = cls.context["variable"]

        def f_modified():
            return L_operator(
                cls.context["solution_exact_expr"], variable
            ) - L_operator(cls.fsys.get_basic_zero(), variable)

        def G_func(i: int, j: int):
            g = lambdify(
                variable,
                L_operator(cls.fsys.get_function(i), variable)
                * L_operator(cls.fsys.get_function(j), variable),
                "numpy",
            )
            return cls.__integration(g, tolerance)

        def L_func(j: int):
            l = lambdify(
                variable,
                f_modified() * L_operator(cls.fsys.get_function(j), variable),
                "numpy",
            )
            return cls.__integration(l, tolerance)

        matrix = np.matrix([[G_func(i, j) for i in range(n)] for j in range(n)])
        vector = np.array([L_func(j) for j in range(n)])
        return matrix, vector

    @classmethod
    def __integration(
        cls, f: Callable[[float], float], tolerance: float = 1e-3
    ) -> float:
        cls.integral.integrand = f
        if cls.strategy is not None and cls.formula is not None:
            return cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        return integrate.quad(cls.integral.integrand, *cls.integral.borders)[0]


class BubnovGalerkin(VariationalProective):
    @classmethod
    def solve(cls, context: dict, n: int, tolerance: float = 1e-3):
        cls.context = context
        cls.constants = context["constants"]
        cls.integral.borders = context["borders"]
        matrix, vector = cls.__build_system(n, tolerance)
        coeficients = np.linalg.solve(matrix, vector)
        error = vector - np.dot(matrix, coeficients)

        def approximation(x: float) -> float:
            return lambdify(
                cls.context["variable"],
                cls.fsys.get_basic_zero()
                + sum([coeficients[i] * cls.fsys.get_function(i) for i in range(n)]),
                "numpy",
            )(x)

        return approximation, error

    @classmethod
    def __build_system(cls, n, tolerance=1e-3):
        variable = cls.context["variable"]
        L_operator = cls.context["L"]

        def f_modified():
            return L_operator(
                cls.context["solution_exact_expr"], variable
            ) - L_operator(cls.fsys.get_basic_zero(), variable)

        def L_phi(i: int, j: int):
            return lambdify(
                variable,
                L_operator(cls.fsys.get_function(i), variable)
                * cls.fsys.get_function(j),
                "numpy",
            )

        def rhs(i: int):
            return lambdify(variable, f_modified() * cls.fsys.get_function(i), "numpy")

        matrix = np.matrix(
            [[cls.__integration(L_phi(i, j)) for i in range(n)] for j in range(n)]
        )
        vector = np.array([cls.__integration(rhs(i)) for i in range(n)])
        return matrix, vector

    @classmethod
    def __integration(
        cls, f: Callable[[float], float], tolerance: float = 1e-3
    ) -> float:
        cls.integral.integrand = f
        if cls.strategy is not None and cls.formula is not None:
            return cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        return integrate.quad(cls.integral.integrand, *cls.integral.borders)[0]

