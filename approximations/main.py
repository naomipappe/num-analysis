from numpy import sin, e, cos, pi

import funsys as fs
from quadraticapproximation import QuadraticApproximation
from legandre import LegendreApproximation
from discrete import DiscreteApproximation
from spline import Spline
import util

#
# def f(x, A: float = 1, w: float = 1):
#     return A * x * sin(w * x ** 2)


def g(x, A=1, w=1):
    return e ** (A * cos(w * x))


# def d(x, A=1, w=1):
#     return A*cos(w*x)*(e**x)


if __name__ == "__main__":
    a = 0
    b = 4
    approximated = g
    verbose = True

    n = int(input("Введите степень полинома: "))

    # trigonometric = QuadraticApproximation(a, b, approximated, fs.TrigonometricSystem())
    exponential = QuadraticApproximation(a, b, approximated, fs.ExponentialSystem())
    # polynomial = QuadraticApproximation(a, b, f, fs.PolynomialSystem())
    legandre = LegendreApproximation(a, b, approximated)
    discrete = DiscreteApproximation(a, b, approximated)
    spline = Spline(a, b, approximated, rho=10 ** 5)

    # Qn_trig = trigonometric.get_mean_quadratic_approximation(n, verbose=verbose)
    # title = "Trigonometric continuous approximation"
    # util.plot_approximation(a, b, title, f=approximated, phi=Qn_trig)
    #
    # print()

    Qn_exp = exponential.get_mean_quadratic_approximation(n, verbose=verbose)
    title = "Exponential continuous approximation"
    util.plot_approximation(a, b, title, f=approximated, phi=Qn_exp)

    print()

    # Qn_polynomial = polynomial.get_mean_quadratic_approximation(n, verbose=verbose)
    # title = "Polynomial continuous approximation"
    # util.plot_approximation(a, b, title, f=approximated, phi=Qn_polynomial)
    #
    # print()

    n = int(input("Введите кол-во узлов для нахождения ЭНСКП по системе многочленов Лежандра: "))
    Qn_legandre = legandre.get_legendre_approximation(n, True)
    title = "Legandre continuous approximation"
    util.plot_approximation(a, b, title, f=approximated, phi=Qn_legandre)

    print()

    m = int(input("Введите кол-во узлов: "))
    Pm = discrete.get_mean_quadratic_approximation(m, verbose=verbose)
    title = "Polynomial discrete approximation"
    util.plot_approximation(a, b, title, approximated, Pm)
    
    print()
    
    m = int(input("Введите кол-во узлов: "))
    s = spline.get_spline(m, verbose=True)
    title = "Spline interpolation"
    util.plot_approximation(a, b, title, f=approximated, phi=s)
    quit()
