# f(x)=Ax*sin(w*x^2), -pi/2 <=x<=3pi/2 A0=1, w0=1
# a=-pi/2
# b=3pi/2
from math import pi
from math import sin
import interpolation as ip
import matplotlib.pyplot as pp
import numpy as np


def f(x, A=1, omega=1):
    return A*x*sin(omega*x**2)


def graph(x, y, labels, title):
    fig, plot1 = pp.subplots()
    plot1.plot(x, y)
    plot1.set(xlabel=labels[0], ylabel=labels[1],
              title=title)
    plot1.grid()
    fig.savefig(title+".png")
    pp.show()


a = -pi/2
b = 3*pi/2
n = 50
x_vals = np.linspace(a, b, 10**3)
y_vals = np.vectorize(f)(x_vals)
y0 = min(y_vals)+2*(max(y_vals)-min(y_vals))/3
p_chebyshev_vals = [ip.interpolate_newton_chebyshew(
    f, x_vals[i], a, b, 40) for i in range(len(x_vals))]
p_equidistant_vals = [ip.interpolate_newton_equidistant(
    f, x_vals[i], a, b, n) for i in range(len(x_vals))]
delta_equidistant = p_equidistant_vals-y_vals
delta_chebyshev = p_chebyshev_vals-y_vals
omega_equidistant = [
    ip.omega(x_vals[i], a, b, n, ip.equidistant_nodes) for i in range(len(x_vals))]

omega_chebyshev = [ip.omega(x_vals[i], a, b, n, ip.chebyshev_nodes)
                   for i in range(len(x_vals))]

graph(x_vals, y_vals, ["x", "y"], "f(x)")
graph(x_vals, p_equidistant_vals, ["x", "P(x)"], "P(x)_equidistant")
graph(x_vals, delta_equidistant, ["x", "f(x)-P(x)"], "P(x)_equidistant_delta")
graph(x_vals, omega_equidistant, ["x", "omega(x)"], "omega_equidistant")
graph(x_vals, p_chebyshev_vals, ["x", "P(x)"], "P(x)_chebyshev")
graph(x_vals, delta_chebyshev, ["x", "f(x)-P(x)"], "P(x)_chebyshev_delta")
graph(x_vals, omega_chebyshev, ["x", "omega(x)"], "omega_chebyshev")
x=ip.reverse_Newton(f,y0,a,b,n)
print(x)
print("real value:",f(x))
print("desired:",f(2.63600958))