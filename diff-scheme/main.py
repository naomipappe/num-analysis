from typing import Callable, Iterable
from scheme.diff_scheme import MonotonicScheme, IntegroInterpolationScheme
from matplotlib import pyplot as plt
from numpy import linspace
from sympy import lambdify
from sympy.abc import symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
)

transformations = standard_transformations + (implicit_multiplication,)

# region Assignment-specific constants
variable = symbols("x")
BORDER_LEFT, BORDER_RIGHT = 1, 3
m1, m2, m3 = 1, 2, 1
p1, p2, p3 = 2, 1, 2
q1, q2, q3 = 1, 1, 1
k1, k2, k3 = 1, 1, 1
beta = -6
delta = 3
# endregion
# region Assignment-specific functions
solution_exact_expression = parse_expr(f'{m1}*sin({m2}*x)+{m3}', evaluate=True)
solution_exact_expression_dx = solution_exact_expression.diff(variable)

k_expression = parse_expr(f'{k1}*(x**{k2})+{k3}', evaluate=True)

p_expression = parse_expr(f'{p1} * (x ** {p2}) + {p3}', evaluate=True)
# p_expression = parse_expr('0', evaluate=True)
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


def plotter(x: list, precise_solution: Callable[[float], float], approximation: Callable[[float], float] or list,
            save: bool = False, name: str = "result"):
    plt.clf()
    plt.plot(x, precise_solution(x), "r")
    if isinstance(approximation, Iterable):
        approx = approximation
    else:
        approx = [approximation(node) for node in x]
    plt.plot(x, approx, "b")
    plt.fill_between(x, precise_solution(x), approx, color="yellow", alpha="0.5")
    if save:
        plt.savefig(f"{name}.png")
    else:
        plt.show()


def main():
    alpha = -k(BORDER_LEFT)
    gamma = k(BORDER_RIGHT)
    n = 50
    nodes = linspace(BORDER_LEFT, BORDER_RIGHT, n)
    scheme = MonotonicScheme((BORDER_LEFT, BORDER_RIGHT), alpha, beta, gamma, delta, mu_1, mu_2, k, p, q)
    approximation = scheme.solve(n, differential_operator(solution_exact_expression))

    print("-------------------------------------------")
    print("x_i  | Справжній  | Наближений | Відхилення")
    print("-------------------------------------------")

    outfile = open(f'result_{n}.csv', 'w+')
    outfile.write('Node,Exact Solution,Approximation,Error\n')
    for i in range(len(nodes)):
        print(
            f"{nodes[i]:.2f} | {solution_exact(nodes[i]):10.7f} | {approximation[i]:10.7f} "
            f"| {abs(solution_exact(nodes[i]) - approximation[i]):10.7f}")
        outfile.write(
            f"{nodes[i]:.2f},{solution_exact(nodes[i]):10.7f},{approximation[i]:10.7f},"
            f"{abs(solution_exact(nodes[i]) - approximation[i]):10.7f}\n")
    outfile.close()
    
    print("------------------------------------------")

    plotter(nodes, solution_exact, approximation, save=True, name=f'result_{n}')


main()
