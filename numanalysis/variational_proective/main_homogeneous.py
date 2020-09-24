from numpy import sin, cos, pi, linspace
from numpy.linalg import norm
from sympy.abc import symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
)

transformations = standard_transformations + (implicit_multiplication,)
from functional.fun_sys import BasisFunction, AlternativeBasis, AnotherSystem
from sympy import lambdify
from methods.methods import Ritz, Collocation, LeastSquares, BubnovGalerkin
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
from scipy import integrate

variable = symbols("x")
a, b = 2, 5
constants = {
    "b1": 1,
    "b2": 2,
    "b3": 1,
    "k1": 6,
    "k2": 3,
    "c1": 1,
    "c2": 2,
    "c3": 1,
    "p1": 2,
    "p2": 1,
    "d1": 1,
    "d2": 1,
    "d3": 1,
    "q1": 1,
    "q2": 0,
    "a1": 6,
    "a2": 3,
    "a3": 1,
    "a4": 1,
    "n1": 2,
    "n2": 1,
    "n3": 4,
}
constants["alpha"] = (
    (constants["a1"] * (a ** (constants["n1"])))
    + (constants["a2"] * (a ** (constants["n2"])))
    + (constants["a3"] * (a ** (constants["n3"])))
    + constants["a4"]
)
constants["beta"] = (
    (constants["a1"] * constants["n1"] * (a ** (constants["n1"] - 1)))
    + (constants["a2"] * constants["n2"] * (a ** (constants["n2"] - 1)))
    + (constants["a3"] * constants["n3"] * (a ** (constants["n3"] - 1)))
)
constants["gamma"] = (
    (constants["a1"] * (b ** (constants["n1"])))
    + (constants["a2"] * (b ** (constants["n2"])))
    + (constants["a3"] * (b ** (constants["n3"])))
    + constants["a4"]
)

constants["delta"] = -(
    (constants["a1"] * constants["n1"] * (b ** (constants["n1"] - 1)))
    + (constants["a2"] * constants["n2"] * (b ** (constants["n2"] - 1)))
    + (constants["a3"] * constants["n3"] * (b ** (constants["n3"] - 1)))
)

solution_exact_expression = parse_expr(
    f'({constants["a1"]}*x**({constants["n1"]}))+({constants["a2"]}*x**({constants["n2"]})) +\
         ({constants["a3"]}*x**({constants["n3"]}))+{constants["a4"]}',
    evaluate=True,
)
solution_exact_expression_dx = solution_exact_expression.diff(variable)
solution_exact_expression_d2x = solution_exact_expression.diff(variable, 2)

k_expression = parse_expr(
    f'{constants["b1"]}*(x**{constants["k1"]})+{constants["b2"]}*(x**{constants["k2"]}) +\
         {constants["b3"]}',
    evaluate=True,
)
k_expresion_dx = k_expression.diff(variable)

p_expression = parse_expr(
    f'{constants["c1"]} * (x ** {constants["p1"]}) + {constants["c2"]}*(x**{constants["p2"]}) +\
         {constants["c3"]}',
    evaluate=True,
)  # comment out for Ritz
# p_expression = parse_expr("0", evaluate=True)  # Uncomment for Ritz

q_expression = parse_expr(
    f'{constants["d1"]} * (x ** {constants["q1"]}) + {constants["d2"]} * (x**{constants["q2"]}) +\
         {constants["d3"]}',
    evaluate=True,
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
    return constants["alpha"] * solution_exact_dx(a) - constants[
        "beta"
    ] * solution_exact(a)


def mu_2() -> float:
    return constants["gamma"] * solution_exact_dx(b) + constants[
        "delta"
    ] * solution_exact(b)


def L_operator(u, variable):
    return (
        u.diff(variable) * (-k_expression + p_expression)
        - k_expression * u.diff(variable, 2)
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
    n_Ritz = 5
    n_Bubnov = 5

    functional_system = AlternativeBasis(context)
    nodes = linspace(a, b, 50, endpoint=True)

    # Ritz.set_functional_system(functional_system)
    # Ritz.set_integration_method(SimpsonsRule, RungeStrategy)
    # approximation_Ritz, error_Ritz = Ritz.solve(context, n_Ritz, 1e-6)

    # plotter(
    #     nodes, solution_exact, approximation_Ritz, save=True, name=f"ritz_{n_Ritz}"
    # )
    # print("Вектор невязки(Метод Ритца):", error_Ritz)
    # print("Норма вектора невязки(Метод Ритца):", norm(error_Ritz))
    BubnovGalerkin.set_functional_system(functional_system)
    BubnovGalerkin.set_integration_method(SimpsonsRule, RungeStrategy)
    approximation_Bubnov, error_Bubnov = BubnovGalerkin.solve(context, n_Bubnov, 1e-6)

    plotter(nodes, solution_exact, approximation_Bubnov, save=True,name=f'bubnov_{n_Bubnov}')
    print("Вектор невязки(Метод Бубнова - Галёркина):", error_Bubnov)
    print("Норма вектора невязки(Метод Бубнова - Галёркина):", norm(error_Bubnov))


if __name__ == "__main__":
    main()