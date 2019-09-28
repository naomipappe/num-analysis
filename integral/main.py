import integrate as intgr
import integration_formulas as formulas
import integration_strategy as strategy
from numpy import inf, pi, sqrt


#
# def upper_border_approximation_test_simpson(eps: float) -> float:
#     return ceil(2 / eps)


def upper_border_approximation(eps: float) -> float:
    upper_border = 1 / 3 * ((9 * eps ** 2 + 3 * sqrt(9 * eps ** 4 + 16 * eps ** 2) + 8) ** (1 / 3) / eps + 4 / (
                eps * (9 * eps ** 2 + 3 * sqrt(9 * eps ** 4 + 16 * eps ** 2) + 8)) + 2 / eps)
    print("Приближение верхнего предела интегрирования: ", upper_border)
    return upper_border


# def upper_border_approximation_test_trapezoidal(eps: float) -> float:
#     return ceil(-log(eps / 2))


def g(x):
    return (x ** 2 + 1) / (x ** 4 + 1)


def d2g(x):
    return 2 * (3 * (x ** 8) + 10 * (x ** 6) - 12 * (x ** 4) - 6 * (x ** 2) + 1) / ((x ** 4 + 1) ** 3)


def main(borders: tuple, integrand, integrand_nth_derivative, tolerance):
    integral = intgr.Integral()
    if borders[1] == inf:
        integral.borders = borders[0], upper_border_approximation(tolerance)
    else:
        integral.borders = borders
    integral.integrand = integrand
    integral.integrand_nth_derivative = integrand_nth_derivative
    formula = formulas.MeanRectangleFormula
    runge_value = integral.integrate(tolerance, strategy.AprioriEstimationStrategy, formula)
    print('I =', runge_value + tolerance / 2)
    print('True value =', pi / sqrt(2))
    adaptive_value = integral.integrate(tolerance, strategy.RungeStrategy, formula)
    print('I =', adaptive_value + tolerance / 2)
    print('True value =', pi / sqrt(2))


if __name__ == '__main__':
    main((0, inf), g, d2g, 1e-3)
