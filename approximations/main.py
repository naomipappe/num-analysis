from numpy import sin, pi, e, cos

import funsys as fs
from quadraticapproximation import QuadraticApproximation
import util
from discrete import DiscreteApproximation
from spline import Spline
from legandre import LegandreApproximation


def f(x, A: float = 1, w: float = 1):
    return A * x * sin(w * x**2)


def g(x, A = 1, w = 1):
    return e**(A * cos(w * x))


if __name__ == "__main__":
    a = 0
    b = 4
    approximated = g
    verbose = True
    try:
        n = int(input("Введите степень полинома: "))
    except ValueError:
        n = 5
        print("Ошибка ввода, используем стандартное значение")
    trigonometric = QuadraticApproximation(a, b, approximated, fs.TrigonometricSystem())
    exponential = QuadraticApproximation(a, b, approximated, fs.ExponentialSystem())
    discrete = DiscreteApproximation(a, b, approximated)
    spline = Spline(a, b, approximated, rho=10**3)
    Qn_trig = trigonometric.get_mean_quadratic_approximation(n, verbose=verbose)
    title = "Trigonometric continuous approximation"
    util.plot_approximation(a, b, title, f=approximated, phi=Qn_trig)
    Qn_exp = exponential.get_mean_quadratic_approximation(n, verbose=verbose)
    title = "Exponential continuous approximation"
    util.plot_approximation(a, b, title, f=approximated, phi=Qn_exp)
    m = int(input("Введите кол-во узлов: "))
    Pm = discrete.get_mean_quadratic_approximation(m)
    title = "Polynomial discrete approximation"
    util.plot_approximation(a, b, title, approximated, Pm)
    m = int(input("Введите кол-во узлов: "))
    s = spline.get_spline(m)
    title = "Spline interpolation"
    util.plot_approximation(a, b, title, f=approximated, phi=s)
    legandre = LegandreApproximation(a, b, approximated)
    Qn_legandre = legandre.get_legandre_approximation_continuous(n, True)
    title = "Legandre continuous approximation"
    util.plot_approximation(a, b, title, f=approximated, phi=Qn_legandre)
    quit()
