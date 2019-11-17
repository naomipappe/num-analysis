from numpy import sin, cos, pi, linspace
from numpy.linalg import norm
from sympy.abc import symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
)

transformations = standard_transformations + (implicit_multiplication,)
from functional.fun_sys import BasisFunction
from sympy import lambdify
from methods.methods import Ritz, Collocation, BubnovGalerkin, LeastSquares
from integral.integration_formulas import (
    SimpsonsRule,
    MeanRectangleFormula,
    TrapezoidalFormula,
)
from integral.integration_strategy import (
    RungeStrategy,
    AdaptiveStrategy,
    AprioriEstimationStrategy,
)
from utilities.util import plotter

variable = symbols("x")

a, b = 1, 2

constants = {
    "m1": 1,
    "m2": 2,
    "m3": 1,
    "p1": 2,
    "p2": 1,
    "p3": 2,
    "q1": 1,
    "q2": 1,
    "q3": 1,
    "k1": 1,
    "k2": 1,
    "k3": 1,
    "beta": 6,
    "delta": 3,
}

solution_exact_expression = parse_expr(
    f'{constants["m1"]}*sin({constants["m2"]}*x)+{constants["m3"]}', evaluate=True
)
solution_exact_expression_dx = solution_exact_expression.diff(variable)
solution_exact_expression_d2x = solution_exact_expression.diff(variable, 2)

k_expression = parse_expr(
    f'{constants["k1"]}*(x**{constants["k2"]})+{constants["k3"]}', evaluate=True
)
k_expresion_dx = k_expression.diff(variable)

p_expression = parse_expr(
    f'{constants["p1"]} * (x ** {constants["p2"]}) + {constants["p3"]}'
)

q_expression = parse_expr(
    f'{constants["q1"]} * (x ** {constants["q2"]}) + {constants["q3"]}'
)


def solution_exact(x: float) -> float:
    return lambdify(variable, solution_exact_expression, "numpy")(x)


def solution_exact_dx(x: float) -> float:
    return lambdify(variable, solution_exact_expression_dx, "numpy")(x)


def solution_exact_d2x(x: float) -> float:
    return lambdify(variable, solution_exact_expression_d2x, "numpy")(x)


def k(x: float) -> float:
    return lambdify(variable, k_expression, "numpy")(x)


def dk(x: float) -> float:
    return lambdify(variable, k_expresion_dx, "numpy")(x)


def p(x: float) -> float:
    return lambdify(variable, p_expression, "numpy")


def q(x: float) -> float:
    return lambdify(variable, q_expression, "numpy")


def mu_1() -> float:
    return -k(a) * solution_exact_dx(a) + constants["beta"] * solution_exact(a)


def mu_2() -> float:
    return k(b) * solution_exact_dx(b) + constants["delta"] * solution_exact(b)


def L_operator(u, variable):
    return (
        -(k_expression * (u.diff(variable))).diff(variable)
        + p_expression * (u.diff(variable))
        + q_expression * u
    )


def main():
    context = {
        "variable": variable,
        "borders": (a, b),
        "constants": constants,
        "k(x)": k,
        "dk/dx": dk,
        "p(x)": p,
        "q(x)": q,
        "mu_1": mu_1,
        "mu_2": mu_2,
        "L": L_operator,
        "solution_exact_expr": solution_exact_expression,
    }
    constants["alpha"] = -k(a)
    constants["gamma"] = k(b)
    n_Ritz = 15
    n_Bubnov = 30
    
    functional_system = BasisFunction(context)
    nodes = linspace(a, b, 100, endpoint=True)
    
    Ritz.set_functional_system(functional_system)
    #Ritz.set_integration_method(SimpsonsRule, RungeStrategy)
    approximation_Ritz, error_Ritz = Ritz.solve(context, n_Ritz, 1e-6)
    
    plotter(nodes, solution_exact, approximation_Ritz, save=False)
    print("Вектор невязки(Метод Ритца):", error_Ritz)
    print("Норма вектора невязки(Метод Ритца):", norm(error_Ritz))
    
    BubnovGalerkin.set_functional_system(functional_system)
    #BubnovGalerkin.set_integration_method(SimpsonsRule, RungeStrategy)
    approximation_Bubnov, error_Bubnov = BubnovGalerkin.solve(context, n_Bubnov, 1e-6)
    
    plotter(nodes, solution_exact, approximation_Bubnov, save=False)
    print("Вектор невязки(Метод Бубнова - Галёркина):", error_Bubnov)
    print("Норма вектора невязки(Метод Бубнова - Галёркина):", norm(error_Bubnov))


if __name__ == "__main__":
    main()
