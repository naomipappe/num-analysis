from quadraticapproximation import QuadraticApproximation
import funsys as fs
import spline as s
from numpy import sin, pi, e, cos
from legandre import LegandreApproximation


def f(x, A = 1, w = 1):
    return x*sin(x**2)


def g(x, A = 2, w = 1):
    return e**(A * cos(w * x))


if __name__ == "__main__":
    a = pi/2
    b = 3*pi/2
    n = int(input("Введите степень полинома: "))
    trigonometric = QuadraticApproximation(a, b, f, fs.TrigonometricSystem())
    exponential = QuadraticApproximation(a, b, f, fs.ExponentialSystem())
    spline = s.Spline(a, b, f)
    # legandre = LegandreApproximation(a, b, f)
    # legandre.plot_approximation(n, 'c')
    # legandre.plot_approximation(n, 'd')
    trigonometric.plot_approximation(n, 'c')
    exponential.plot_approximation(n, 'c')
    trigonometric.plot_approximation(n, 'd')
    spline.plot_spline(n*10)
