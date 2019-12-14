from scipy import integrate
from sympy import lambdify
from sympy.abc import symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
)
from scheme.diff_scheme import FiniteElementDiffScheme, IntegroInterpolationScheme
from matplotlib import pyplot as plt
from typing import Callable
from numpy import linspace
from collections.abc import Iterable
transformations = standard_transformations + (implicit_multiplication,)
# region constants and conditions
variable = symbols("x")
a, b = 1, 3
constants = {
    "b1": 1,
    "b2": 0,
    "b3": 0,
    "k1": 0,
    "k2": 1,
    "c1": 2,
    "c2": 0,
    "c3": 1,
    "p1": 2,
    "p2": 0,
    "d1": 1,
    "d2": 2,
    "d3": 2,
    "q1": 0,
    "q2": 1,
    "a1": 3,
    "a2": 1,
    "a3": 1,
    "a4": 1,
    "n1": 1,
    "n2": 1,
    "n3": 2,
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
    evaluate=True)
solution_exact_expression_dx = solution_exact_expression.diff(variable)
solution_exact_expression_d2x = solution_exact_expression.diff(variable, 2)

k_expression = parse_expr(
    f'{constants["b1"]}*(x**{constants["k1"]})+{constants["b2"]}*(x**{constants["k2"]}) +\
         {constants["b3"]}',
    evaluate=True,
)
k_expresion_dx = k_expression.diff(variable)

# p_expression = parse_expr(
#     f'{constants["c1"]} * (x ** {constants["p1"]}) + {constants["c2"]}*(x**{constants["p2"]}) +\
#          {constants["c3"]}',
#     evaluate=True,
# )  # comment out for Ritz
p_expression = parse_expr("0", evaluate=True)  # Uncomment for Ritz

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
    return lambdify(variable, p_expression, "numpy")(x)


def q(x: float) -> float:
    return lambdify(variable, q_expression, "numpy")(x)


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
        (-k_expression*u.diff(variable)).diff(variable) +
        q_expression*solution_exact_expression
    )
# endregion


def plotter(
    x: list,
    precise_solution: Callable[[float], float],
    approximation: Callable[[float], float] or list,
    save: bool = False,
    name: str = "result",
):
    plt.clf()
    plt.plot(x, precise_solution(x), "r")
    if isinstance(approximation, Iterable):
        approx = approximation
    else:
        approx = [approximation(node) for node in x]
    plt.plot(x, approx, "b")
    plt.fill_between(
        x, precise_solution(x), approx, color="yellow", alpha="0.5"
    )
    if save:
        plt.savefig(f"{name}.png")
    else:
        plt.show()


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
    n = 10
    nodes, step = linspace(a, b, n, retstep=True)
    scheme = IntegroInterpolationScheme(context)
    approximation = scheme.solve(nodes, step)

    print("-------------------------------------------")
    print("x_i  | Справжній  | Наближений | Відхилення")
    print("-------------------------------------------")
    
    outfile = open('result.csv', 'w+')
    outfile.write('Node,Exact Solution,Approximation,Error\n')
    for i in range(len(nodes)):
        print(f"{nodes[i]:.2f} | {solution_exact(nodes[i]):10.7f} | {approximation[i]:10.7f} | {abs(solution_exact(nodes[i]) - approximation[i]):10.7f}")
        outfile.write(
            f"{nodes[i]:.2f},{solution_exact(nodes[i]):10.7f},{approximation[i]:10.7f},{abs(solution_exact(nodes[i]) - approximation[i]):10.7f}\n")
    outfile.close()
    
    print("------------------------------------------")

    plotter(nodes, solution_exact, approximation,
            save=True, name=f'result_{n}')


main()
