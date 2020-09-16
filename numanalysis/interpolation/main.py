# f(x)=Ax*sin(w*x^2), -pi/2 <=x<=3pi/2 A0=1, w0=1
# a=-pi/2
# b=3pi/2
import interpolation as ip
import matplotlib.pyplot as pp
import numpy as np


def f(arg,A=1,omega=1):
    return np.cos(omega*arg)*A*np.e**arg


def graph(x_, y_, labels, title):
    fig, plot1 = pp.subplots()
    plot1.plot(x_, y_)
    plot1.set(xlabel=labels[0], ylabel=labels[1],
              title=title)
    plot1.grid()
    fig.savefig(title+".png")
    pp.show()


a = 1
b = 4
n = 5
x_values = np.linspace(a, b, 10 ** 4)
y_values = np.vectorize(f)(x_values)
y0 = min(y_values) + 2 * (max(y_values) - min(y_values)) / 3
p_chebyshev_values = np.array([ip.interpolate_newton_chebyshew(
    f, x_values[i], a, b, n) for i in range(len(x_values))])
p_equidistant_values = np.array([ip.interpolate_newton_equidistant(
    f, x_values[i], a, b, n) for i in range(len(x_values))])
delta_equidistant = p_equidistant_values - y_values
delta_chebyshev = p_chebyshev_values - y_values
omega_equidistant = [
    ip.omega(x_values[i], a, b, n, ip.equidistant_nodes) for i in range(len(x_values))]

omega_chebyshev = [ip.omega(x_values[i], a, b, n, ip.chebyshev_nodes)
                   for i in range(len(x_values))]

graph(x_values, y_values, ["x", "y"], "f(x)")
graph(x_values, p_equidistant_values, ["x", "P(x)"], "P(x)_equidistant")
graph(x_values, delta_equidistant, ["x", "f(x)-P(x)"], "P(x)_equidistant_delta")
graph(x_values, omega_equidistant, ["x", "omega(x)"], "omega_equidistant")
graph(x_values, p_chebyshev_values, ["x", "P(x)"], "P(x)_chebyshev")
graph(x_values, delta_chebyshev, ["x", "f(x)-P(x)"], "P(x)_chebyshev_delta")
graph(x_values, omega_chebyshev, ["x", "omega(x)"], "omega_chebyshev")

x = ip.reverse_Newton(f, y0, n)
print("y* =", y0)
print("x* =", x)
print("f(x*) =", f(x))
print("Delta =", abs(f(x)-y0))
