from typing import Type, Tuple, Callable

from numpy import array, dot, linalg, linspace
from scipy import integrate
from sympy import lambdify

from numanalysis.variational_proective.functional.fun_sys import FunctionalSystem


class VariationalProective:
    _functional_system: Type[FunctionalSystem] = None

    @classmethod
    def approximation(cls, n: int, equation_rhs, differential_operator):
        __coefficients, error = cls.coefficients(n, equation_rhs, differential_operator)

        def f(x):
            return lambdify(
                cls._functional_system.variable,
                cls._functional_system.get_basic_zero() + sum(
                    [__coefficients[i] * cls._functional_system.get_function(i)
                     for i in range(n)]), 'numpy')(x)

        return f, error

    @classmethod
    def coefficients(cls, n, equation_rhs, differential_operator) -> Tuple[Callable[[float], float], list]:
        raise NotImplementedError

    @classmethod
    def set_functional_system(cls, functional_system: Type[FunctionalSystem]):
        cls._functional_system = functional_system


class Ritz(VariationalProective):
    @classmethod
    def coefficients(cls, n, equation_rhs, differential_operator):
        matrix, vector = cls.__build_system(n, equation_rhs, differential_operator)
        _coefficients = linalg.solve(matrix, vector)
        error = vector - dot(matrix, _coefficients)
        return _coefficients, error

    @classmethod
    def __build_system(cls, n: int, equation_rhs, differential_operator):
        def rhs_homogeneous():
            return equation_rhs - differential_operator(cls._functional_system.get_basic_zero())

        def matrix_element(i: int, j: int) -> float:
            return integrate.quad(
                lambdify(cls._functional_system.variable, differential_operator(
                    cls._functional_system.get_function(i)) * cls._functional_system.get_function(j), "numpy", ),
                *cls._functional_system.borders)[0]

        def vector_element(i: int) -> float:
            return integrate.quad(
                lambdify(cls._functional_system.variable, rhs_homogeneous() * cls._functional_system.get_function(i),
                         "numpy"), *cls._functional_system.borders)[0]

        matrix = array(
            [[matrix_element(i, j) for i in range(n)] for j in range(n)]
        )

        vector = array([vector_element(i) for i in range(n)])

        return matrix, vector


class Collocation(VariationalProective):
    __nodes = None

    @classmethod
    def coefficients(cls, n: int, equation_rhs, differential_operator):
        matrix, vector = cls.__build_system(n, equation_rhs, differential_operator)
        _coefficients = linalg.solve(matrix, vector)
        error = vector - dot(matrix, _coefficients)
        return _coefficients, error

    @classmethod
    def __build_system(cls, n: int, equation_rhs, differential_operator):
        def rhs_homogeneous(x: float) -> float:
            return lambdify(cls._functional_system.variable,
                            equation_rhs - differential_operator(cls._functional_system.get_basic_zero()),
                            'numpy')(x)

        def make_matrix_element(i: int, j: int) -> float:
            return lambdify(cls._functional_system.variable,
                            differential_operator(cls._functional_system.get_function(j)), 'numpy')(cls.__nodes[i])

        cls.__nodes = linspace(*cls._functional_system.borders, n, endpoint=True)

        matrix = array(
            [[make_matrix_element(i, j) for j in range(len(cls.__nodes))] for i in range(len(cls.__nodes))]
        )
        vector = array([rhs_homogeneous(cls.__nodes[i]) for i in range(n)])

        return matrix, vector


class LeastSquares(VariationalProective):
    @classmethod
    def coefficients(cls, n, equation_rhs, differential_operator) -> Tuple[Callable[[float], float], list]:
        matrix, vector = cls.__build_system(n, equation_rhs, differential_operator)
        _coefficients = linalg.solve(matrix, vector)
        error = vector - dot(matrix, _coefficients)
        return _coefficients, error

    @classmethod
    def __build_system(cls, n: int, equation_rhs, differential_operator):
        def rhs_homogeneous():
            return equation_rhs - differential_operator(cls._functional_system.get_basic_zero())

        def make_matrix_element(i: int, j: int) -> float:
            return integrate.quad(lambdify(cls._functional_system.variable,
                                           differential_operator(
                                               cls._functional_system.get_function(i)) *
                                           differential_operator(
                                               cls._functional_system.get_function(j)), "numpy"),
                                  *cls._functional_system.borders)[0]

        def make_vector_element(j: int) -> float:
            return integrate.quad(lambdify(cls._functional_system.variable,
                                           rhs_homogeneous() * differential_operator(
                                               cls._functional_system.get_function(j)),
                                           'numpy'), *cls._functional_system.borders)[0]

        matrix = array([[make_matrix_element(i, j) for i in range(n)] for j in range(n)])
        vector = array([make_vector_element(j) for j in range(n)])
        print(matrix)
        return matrix, vector


class BubnovGalerkin(VariationalProective):
    @classmethod
    def coefficients(cls, n, equation_rhs, differential_operator) -> Tuple[Callable[[float], float], list]:
        matrix, vector = cls.__build_system(n, equation_rhs, differential_operator)
        __coefficients = linalg.solve(matrix, vector)
        error = vector - dot(matrix, __coefficients)
        return __coefficients, error

    @classmethod
    def __build_system(cls, n: int, equation_rhs, differential_operator):
        def rhs_homogeneous():
            return equation_rhs - differential_operator(cls._functional_system.get_basic_zero())
        def make_matrix_element(i: int, j: int):
            return integrate.quad(lambdify(
                cls._functional_system.variable,
                differential_operator(cls._functional_system.get_function(i)) * cls._functional_system.get_function(j),
                "numpy"), *cls._functional_system.borders)[0]

        def make_vector_element(i: int):
            return integrate.quad(
                lambdify(cls._functional_system.variable, rhs_homogeneous() * cls._functional_system.get_function(i),
                         "numpy"), *cls._functional_system.borders)[0]

        matrix = array(
            [[make_matrix_element(i, j) for i in range(n)] for j in range(n)]
        )
        vector = array([make_vector_element(i) for i in range(n)])

        return matrix, vector
