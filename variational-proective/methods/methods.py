from integral.integrate import Integral
from integral.integration_strategy import IntegrationStrategy
from integral.integration_formulas import QuadraticFormula
from typing import Type, Tuple, Callable
from sympy import lambdify
from functional.fun_sys import FunctionalSystem
import numpy as np
from scipy import integrate

# TODO speed up Ritz


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
        cls.constants = context["constants"]
        if tolerance is None:
            tolerance = 1e-3
        cls.integral.borders = context["borders"]
        matrix, vector = cls.__build_system(n, tolerance)
        coeficients = np.linalg.solve(matrix, vector)
        error = vector - np.dot(matrix, coeficients)
        print(np.linalg.norm(error))
        print(coeficients)

        def approximation(x: float) -> float:
            return lambdify(  # TODO DRY principle
                cls.context["variable"],
                cls.fsys.get_basic_zero()
                + sum([coeficients[i] * cls.fsys.get_function(i) for i in range(n)]),
                "numpy",
            )(x)

        return approximation

    @classmethod
    def __build_system(cls, n: int, tolerance):
        matrix = np.matrix(
            [[cls.G_func(i, j, tolerance) for i in range(n)] for j in range(n)]
        )
        vector = np.array([cls.L_func(i, tolerance) for i in range(n)])
        return matrix, vector

    @classmethod
    def G_func(cls, i: int, j: int, tolerance: float) -> float:
        k = cls.context["k(x)"]
        q = cls.context["q(x)"]
        variable = cls.context["variable"]

        def g(x: float) -> float:
            return k(x) * lambdify(
                variable,
                cls.fsys.get_derrivative(i, 1) * cls.fsys.get_derrivative(j, 1),
                "numpy",
            )(x)
            +q(x) * lambdify(
                variable, cls.fsys.get_function(i) * cls.fsys.get_function(j), "numpy",
            )(x)

        res = cls.__integration(g, tolerance)
        res += (
            cls.constants["beta"]
            * lambdify(cls.context["variable"], cls.fsys.get_function(i), "numpy")(
                cls.context["borders"][0]
            )
            * lambdify(cls.context["variable"], cls.fsys.get_function(j), "numpy")(
                cls.context["borders"][0]
            )
        )
        res += (
            cls.constants["gamma"]
            * lambdify(cls.context["variable"], cls.fsys.get_function(i), "numpy")(
                cls.context["borders"][1]
            )
            * lambdify(cls.context["variable"], cls.fsys.get_function(j), "numpy")(
                cls.context["borders"][1]
            )
        )
        return res

    @classmethod
    def L_func(cls, i: int, tolerance: float) -> float:
        def f_modified(x: float) -> float:
            return lambdify(
                cls.context["variable"],
                cls.context["L"](
                    cls.context["solution_exact_expr"], cls.context["variable"]
                ).simplify()
                - cls.context["L"](
                    cls.fsys.get_basic_zero(), cls.context["variable"]
                ).simplify(),
                "numpy",
            )(x)

        def v(x: float, i: int) -> float:
            return lambdify(cls.context["variable"], cls.fsys.get_function(i), "numpy")(
                x
            )

        return cls.__integration(lambda x: f_modified(x) * v(x, i), tolerance)

    @classmethod
    def __integration(
        cls, f: Callable[[float], float], tolerance: float = 1e-3
    ) -> float:
        cls.integral.integrand = f
        # return cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        return integrate.quad(cls.integral.integrand, *cls.integral.borders)[0]


class Collocation(VariationalProective):
    __nodes = None

    @classmethod
    def solve(cls, context, n, tolerance=1e-3):
        cls.context = context
        matrix, vector = cls.__build_system(context, n)
        coeficients = np.linalg.solve(matrix, vector)

        def approximation(x: float) -> float:
            return lambdify(
                cls.context["variable"],
                cls.fsys.get_basic_zero()
                + sum([coeficients[i] * cls.fsys.get_function(i) for i in range(n)]),
                "numpy",
            )(x)

        return approximation

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
        raise NotImplementedError("Least Squares")

    @classmethod
    def __build_system(cls, n, tolerance=1e-3):
        pass


class BubnovGalerkin(VariationalProective):
    @classmethod
    def solve(cls, context: dict, n: int, tolerance: float = 1e-3):
        cls.context = context
        cls.constants = context["constants"]
        cls.integral.borders = context["borders"]
        matrix, vector = cls.__build_system(n, tolerance)
        coeficients = np.linalg.solve(matrix, vector)

        def approximation(x: float) -> float:
            return lambdify(
                cls.context["variable"],
                cls.fsys.get_basic_zero()
                + sum([coeficients[i] * cls.fsys.get_function(i) for i in range(n)]),
                "numpy",
            )(x)

        return approximation

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
        # return cls.integral.integrate(tolerance, cls.strategy, cls.formula)
        return integrate.quad(cls.integral.integrand, *cls.integral.borders)[0]

