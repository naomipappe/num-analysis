from quadraticapproximation import QuadraticApproximation
import funsys as fs
import spline as s
from numpy import sin, pi


def f(x):
    return x*sin(x**2)


if __name__ == "__main__":
    a = pi/2
    b = 3*pi/2
    n = int(input("Введите степень полинома: "))
    trigonometric = QuadraticApproximation(a, b, f, fs.TrigonometricSystem())
    exponential = QuadraticApproximation(a, b, f, fs.ExponentialSystem())
    spline = s.Spline(a, b, f)
    trigonometric.plot_approximation(n, 'c')
    trigonometric.plot_approximation(n*10, 'd')
    exponential.plot_approximation(n, 'c')
    exponential.plot_approximation(n*20, 'd')
    spline.plot_spline(n*10)
