#x*e^x-x-1; положительный; Ньютона, секущих, релаксации
import math
#import numpy as np


def f(x):
    return x*(math.e**x)-x-1


def d(x):
    return math.e**x+x*math.e**x-1


def input_check(a: float, b: float, x0: float):
    if a >= b:
        raise ValueError(f"Invalid boundaries, a={a}>=b={b}")
    if x0 < a or x0 > b:
        raise ValueError(
            f"Invalid value, x0={x0} is out of boundaries ({a},{b})")


def newton(f, d, x0: float, a: float, b: float, eps: float=1e-5)->float:
    input_check(a, b, x0)

    def step(x):
        return x-f(x)/d(x)
    
    xi = x0
    i = 0
    
    while abs(step(xi)-xi) >= eps:
        i += 1
        xi = step(xi)
    return step(xi)


a = float(input())
b = float(input())
x0 = float(input())
print(newton(f, d, x0, a, b))
