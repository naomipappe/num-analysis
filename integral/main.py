from numpy import inf, ceil, log, pi

import integrate as intgr
import integration_formulas as formulas
import integration_strategy as strategy


def upper_border_approximation_test_simpson(eps: float) -> float:
    return ceil(2 / eps)


def upper_border_approximation_test_rect(eps: float) -> float:
    upper_border = ceil(2.42 / eps)
    print("Приближение верхнего предела интегрирования: ", upper_border)
    return upper_border


def upper_border_approximation_test_trapezoidal(eps: float) -> float:
    return ceil(-log(eps / 2))


def g(x):
    return 1 / (x ** 2 + 4 * x + 13)


def d4g(x):
    return 24 * (2 * x + 4) ** 4 / (x ** 2 + 4 * x + 13) ** 5 - 72 * (2 * x + 4) ** 2 / (
            x ** 2 + 4 * x + 13) ** 4 + 24 / (x ** 2 + 4 * x + 13) ** 3


def main(borders: tuple, integrand, integrand_nth_derivative, tolerance):
    integral = intgr.Integral()
    if borders[1] == inf:
        integral.borders = borders[0], upper_border_approximation_test_simpson(tolerance)
    integral.integrand = integrand
    integral.integrand_nth_derivative = integrand_nth_derivative
    formula = formulas.SimpsonsRule
    apriori_value = integral.integrate(tolerance / 2, strategy.AprioriEstimationStrategy, formula)
    print('I =', apriori_value + tolerance / 2)
    print('True value =', pi / 12)
    adaptive_value = integral.integrate(tolerance / 2, strategy.AdaptiveStrategy, formula)
    print('I =', adaptive_value + tolerance / 2)
    print('True value =', pi / 12)


if __name__ == '__main__':
    main((1, inf), g, d4g, 1e-5)
