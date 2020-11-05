from numanalysis.variational_proective.utilities.util import plotter
from numanalysis.variational_proective.methods.methods import BubnovGalerkin
from sympy import lambdify
from numanalysis.variational_proective.functional.fun_sys import EvenOddBasis, BasisFunction, AnotherSystem
from numpy import linspace
from numpy.linalg import norm
from sympy.abc import symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
)

transformations = standard_transformations + (implicit_multiplication,)

variable = symbols("x")
a, b = 2, 5
b1, b2, b3 = 1, 2, 1
k1, k2 = 6, 3
c1, c2, c3 = 1, 2, 1
p1, p2 = 2, 1
d1, d2, d3 = 1, 1, 1
q1, q2 = 1, 0
a1, a2, a3, a4 = 6, 3, 1, 1
n1, n2, n3 = 2, 1, 4
alpha = a1*(a**n1) + a2*(a**n2) + a3*(a**n3)+a4
beta = a1 * n1 * (a ** (n1 - 1)) + a2 * n2 * \
    (a ** (n2 - 1)) + a3 * n3 * (a ** (n3 - 1))
gamma = a1*(b**n1) + a2*(b**n2) + a3*(b**n3)+a4
delta = a1 * n1 * (b ** (n1 - 1)) + a2 * n2 * \
    (b ** (n2 - 1)) + a3 * n3 * (b ** (n3 - 1))

solution_exact_expression = parse_expr(
    f'({a1}*x**({n1})) + ({a2}*x**({n2})) + ({a3}*x**({n3}))+{a4}',
    evaluate=True,
)
solution_exact_expression_dx = solution_exact_expression.diff(variable)
solution_exact_expression_d2x = solution_exact_expression.diff(variable, 2)

k_expression = parse_expr(
    f'({b1}*(x**{k1})) + ({b2}*(x**{k2})) + {b3}',
    evaluate=True,
)
k_expresion_dx = k_expression.diff(variable)

p_expression = parse_expr(
    f'{c1}*(x ** {p1}) + {c2}*(x**{p2}) + {c3}',
    evaluate=True,
)  # comment out for Ritz
# p_expression = parse_expr("0", evaluate=True)  # Uncomment for Ritz

q_expression = parse_expr(
    f'{d1}*(x ** {q1}) + {d2}*(x**{q2}) + {d3}',
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
    return lambdify(variable, p_expression, "numpy")(x)


def q(x: float) -> float:
    return lambdify(variable, q_expression, "numpy")(x)


def mu_1() -> float:
    return alpha * solution_exact_dx(a) - beta * solution_exact(a)


def mu_2() -> float:
    return gamma * solution_exact_dx(b) - delta * solution_exact(b)


def differential_operator(u):
    return -(k_expression * u.diff(variable)).diff(variable) + p_expression * u.diff(variable) + \
        q_expression * u


def main():
    n_Bubnov = 5
    functional_system = AnotherSystem(
        (a, b), alpha, beta, gamma, delta, k, mu_1, mu_2, variable
    )
    nodes = linspace(a, b, 200, endpoint=True)

    # Ritz.set_functional_system(functional_system)
    # Ritz.set_integration_method(SimpsonsRule, RungeStrategy)
    # approximation_Ritz, error_Ritz = Ritz.solve(context, n_Ritz, 1e-6)

    # plotter(
    #     nodes, solution_exact, approximation_Ritz, save=True, name=f"ritz_{n_Ritz}"
    # )
    # print("Вектор невязки(Метод Ритца):", error_Ritz)
    # print("Норма вектора невязки(Метод Ритца):", norm(error_Ritz))
    BubnovGalerkin.set_functional_system(functional_system)
    approximation_Bubnov, error_Bubnov = BubnovGalerkin.approximation(
        n_Bubnov, differential_operator(solution_exact_expression), differential_operator)

    plotter(nodes, solution_exact, approximation_Bubnov,
            save=False, name=f'bubnov_{n_Bubnov}')
    print("Вектор невязки(Метод Бубнова - Галёркина):", error_Bubnov)
    print("Норма вектора невязки(Метод Бубнова - Галёркина):", norm(error_Bubnov))


if __name__ == "__main__":
    main()
