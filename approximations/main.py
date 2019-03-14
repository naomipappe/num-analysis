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
    a = 0

    b = 4
    n = int(input("Введите степень полинома: "))
    trigonometric = QuadraticApproximation(a, b, g, fs.TrigonometricSystem())
    exponential = QuadraticApproximation(a, b, g, fs.ExponentialSystem())
    spline = s.Spline(a, b, g)
    legandre = LegandreApproximation(a, b, g)
    trigonometric.plot_approximation(n, 'c')
    trigonometric.plot_approximation(5*n, 'd')
    legandre.plot_approximation(n, 'c')
    legandre.plot_approximation(n, 'd')
    spline.plot_spline(n*10)
