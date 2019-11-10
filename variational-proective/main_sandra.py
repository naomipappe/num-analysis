from numpy import sin, cos, pi
from sympy.abc import symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
)

transformations = standard_transformations + (implicit_multiplication,)
from functional.fun_sys import BasisFunction
from sympy import lambdify

variable = symbols("x")

borders = (1, 2)

constants = {
    "m1": 1,
    "m2": 2,
    "m3": 1,
    "m4": 6,
    "m5": 3,
    "p1": 2,
    "p2": 1,
    "p3": 2,
    "q1": 1,
    "q2": 1,
    "q3": 1,
    "k1": 1,
    "k2": 1,
    "k3": 1,
    "alpha_1": 6,
    "alpha_2": 3,
}

solution_exact_expression = parse_expr(
    f'{constants["m1"]}*sin({constants["m2"]}*x)+{constants["m3"]}', evaluate=False
)
solution_exact_expression_dx = solution_exact_expression.diff(variable)
solution_exact_expression_d2x = solution_exact_expression.diff(variable, 2)

k_expression = parse_expr(
    f'{constants["k1"]}*(x**{constants["k2"]})+{constants["k3"]}', evaluate=False
)
k_dx = k_expression.diff(variable)


def solution_exact():
    return solution_exact_expression


def solution_exact_dx():
    return solution_exact_expression_dx


def solution_exact_d2x():
    return solution_exact_expression_d2x


def k():
    return k_expression


def dk():
    return k_dx


def p(x: float) -> float:
    return constants["p1"] * (x ** constants["p2"]) + constants["p3"]


def q(x: float) -> float:
    return constants["q1"] * (x ** constants["q2"]) + constants["q3"]


def f(x: float):
    return (
        -constants["k1"]
        * (x ** constants["k2"])
        * constants["k2"]
        * constants["m1"]
        * constants["m2"]
        * cos(constants["m2"] * x)
        / x
        + (constants["k1"] * (x ** constants["k2"]) + constants["k3"])
        * constants["m1"]
        * (constants["m2"] ** 2)
        * sin(constants["m2"] * x)
        + (constants["p1"] * (x ** constants["p2"]) + constants["p3"])
        * constants["m1"]
        * constants["m2"]
        * cos(constants["m2"] * x)
        + (constants["q1"] * (x ** constants["q2"]) + constants["q3"])
        * (constants["m1"] * sin(constants["m2"] * x) + constants["m3"])
    )


def mu_1():
    return -(
        constants["k1"] * (borders[0] ** constants["k2"]) + constants["k3"]
    ) * constants["m1"] * constants["m2"] * cos(
        constants["m2"] * borders[0]
    ) + constants[
        "alpha_1"
    ] * (
        constants["m1"] * sin(constants["m2"] * borders[0]) + constants["m3"]
    )


def mu_2():
    return (
        constants["k1"] * (borders[1] ** constants["k2"]) + constants["k3"]
    ) * constants["m1"] * constants["m2"] * cos(
        constants["m2"] * borders[1]
    ) + constants[
        "alpha_2"
    ] * (
        constants["m1"] * sin(constants["m2"] * borders[1]) + constants["m3"]
    )


def L(x: float) -> float:
    return (
        (-1)
        * (
            lambdify(variable, dk(), "numpy")(x)
            * lambdify(variable, solution_exact_dx(), "numpy")(x)
            + lambdify(variable, k(), "numpy")(x)
            * lambdify(variable, solution_exact_d2x(), "numpy")(x)
        )
        + p(x) * lambdify(variable, solution_exact_dx(), "numpy")(x)
        + q(x) * lambdify(variable, solution_exact(), "numpy")(x)
    )


def L_Ritz(x: float) -> float:
    return (-1) * (
        lambdify(variable, dk(), "numpy")(x)
        * lambdify(variable, solution_exact_dx(), "numpy")(x)
        + lambdify(variable, k(), "numpy")(x)
        * lambdify(variable, solution_exact_d2x(), "numpy")(x)
    ) + q(x) * lambdify(variable, solution_exact(), "numpy")(x)


def main():
    context = {
        "borders": borders,
        "constants": constants,
        "k(x)": k,
        "dk/dx": dk,
        "p(x)": p,
        "q(x)": q,
        "f(x)": f,
        "mu_1": mu_1,
        "mu_2": mu_2,
        "solution_exact": solution_exact,
        "solution_exact_dx": solution_exact_dx,
        "solution_exact_d2x": solution_exact_d2x,
    }
    n = 10
    tst = BasisFunction(context)

    def L_basis_zero(x: float) -> float:
        return (-1) * (
            lambdify(variable, dk(), "numpy")(x)
            * tst.get_first_derivative_function(0)(x)
            + lambdify(variable, k(), "numpy")(x)
            * tst.get_second_derivative_function(0)(x)
        )
        +p(x) * tst.get_first_derivative_function(0)(x)
        +q(x) * tst.get_function(0)(x)

    def L_Ritz_zero(x: float) -> float:
        return (-1) * (
            lambdify(variable, dk(), "numpy")(x)
            * tst.get_first_derivative_function(0)(x)
            + lambdify(variable, k(), "numpy")(x)
            * tst.get_second_derivative_function(0)(x)
        ) + q(x) * tst.get_function(0)(x)

    def newL(x: float) -> float:
        return L(x) - L_basis_zero(x)
    
    def new_Ritz_l(x:float)->float:
        return L_Ritz(x) - L_Ritz_zero(x)


if __name__ == "__main__":
    main()
