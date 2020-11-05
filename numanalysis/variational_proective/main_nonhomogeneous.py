from numpy import linspace
from scipy import integrate
from sympy import lambdify
from sympy.abc import symbols
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication, )

from numanalysis.variational_proective.functional.fun_sys import BasisFunction
from numanalysis.variational_proective.methods.methods import Ritz, BubnovGalerkin
from numanalysis.variational_proective.utilities.util import plotter

transformations = standard_transformations + (implicit_multiplication,)
variable = symbols("x")

# region assignment specific constants

BORDER_LEFT, BORDER_RIGHT = 1, 2
m1, m2, m3 = 1, 10, 1
p1, p2, p3 = 2, 1, 2
q1, q2, q3 = 1, 1, 1
k1, k2, k3 = 1, 1, 1
beta = -6
delta = 3
# endregion
# region assignment specific functions
solution_exact_expression = parse_expr(f'{m1}*sin({m2}*x)+{m3}', evaluate=True)
solution_exact_expression_dx = solution_exact_expression.diff(variable)

k_expression = parse_expr(f'{k1}*(x**{k2})+{k3}', evaluate=True)

p_expression = parse_expr(f'{p1} * (x ** {p2}) + {p3}', evaluate=True)

q_expression = parse_expr(f'{q1} * (x ** {q2}) + {q3}', evaluate=True)


def solution_exact(x: float) -> float:
    return lambdify(variable, solution_exact_expression, "numpy")(x)


def solution_exact_dx(x: float) -> float:
    return lambdify(variable, solution_exact_expression_dx, 'numpy')(x)


def k(x: float) -> float:
    return lambdify(variable, k_expression, "numpy")(x)


def p(x: float) -> float:
    return lambdify(variable, p_expression, "numpy")(x)


def q(x: float) -> float:
    return lambdify(variable, q_expression, "numpy")(x)


def mu_1() -> float:
    return -k(BORDER_LEFT) * solution_exact_dx(BORDER_LEFT) + beta * solution_exact(BORDER_LEFT)


def mu_2() -> float:
    return k(BORDER_RIGHT) * solution_exact_dx(BORDER_RIGHT) + delta * solution_exact(BORDER_RIGHT)


def differential_operator(u):
    return (-k_expression * u.diff(variable)).diff(variable) + p_expression * u.diff(variable) + \
        q_expression * u

# endregion


def main():
    alpha = -k(BORDER_LEFT)
    gamma = k(BORDER_RIGHT)
    nodes = linspace(BORDER_LEFT, BORDER_RIGHT, 50, endpoint=True)
    # region Ritz
    n_Ritz = 8
    functional_system = BasisFunction(
        (BORDER_LEFT, BORDER_RIGHT), alpha, beta, gamma, delta, k, mu_1, mu_2, variable)
    Ritz.set_functional_system(functional_system)

    approximation_ritz, error_ritz = Ritz.approximation(n_Ritz, differential_operator(solution_exact_expression),
                                                        differential_operator)
    plotter(nodes, solution_exact, approximation_ritz,
            save=False, name=f'Ritz{n_Ritz}')
    norm_err_ritz = \
        integrate.quad(lambda x: (solution_exact(x) - approximation_ritz(x)) ** 2, BORDER_LEFT, BORDER_RIGHT)[0] / \
        (BORDER_RIGHT - BORDER_LEFT)
    print('Отклонение метода Ритца по норме:', norm_err_ritz)
    # endregion

    # region Bubnov
    n_Bubnov = 8
    BubnovGalerkin.set_functional_system(functional_system)

    approximation_bubnov, error_bubnov = BubnovGalerkin.approximation(n_Bubnov,
                                                                      differential_operator(
                                                                          solution_exact_expression),
                                                                      differential_operator)
    norm_err_bubnov = \
        integrate.quad(lambda x: (solution_exact(x) - approximation_bubnov(x)) ** 2, BORDER_LEFT, BORDER_RIGHT)[0] / \
        (BORDER_RIGHT - BORDER_LEFT)

    plotter(nodes, solution_exact, approximation_bubnov,
            save=False, name=f'Bubnov{n_Bubnov}')
    print('Отклонение метода Бубнова-Галёркина по норме:', norm_err_bubnov)
    # endregion


if __name__ == "__main__":
    main()
