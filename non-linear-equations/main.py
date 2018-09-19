#! usr/bin/env python3
# -*- coding: utf-8 -*-
#x*e^x-x-1; положительный; Ньютона, секущих, релаксации
import math
#import numpy as np


def f(x):
    return x*(math.e**x)-x-1


def d(x):
    return math.e**x+x*math.e**x-1


def input_check(a: float, b: float, x0: float):
    if a >= b:
        raise ValueError(f"Invalid boundaries, a={a} >= b ={b}")
    if x0 < a or x0 > b:
        raise ValueError(
            f"Invalid starting point, x0={x0} is out of boundaries ({a},{b})")


def newton(f, d, x0: float, a: float, b: float, logs: bool=True, eps: float=1e-5)->float:
    input_check(a, b, x0)

    def step(x):
        return x-f(x)/d(x)

    xi = x0
    i = 0

    while abs(step(xi)-xi) >= eps:
        i += 1
        if logs:
            print(f"i={i}, xi={xi}, f(xi)={f(xi)}")
        xi = step(xi)
    return step(xi)


def secant(f, x0: float, x1: float, a: float, b: float, logs: bool=True, eps: float=1e-5)->float:
    input_check(a, b, x0)
    input_check(a, b, x1)
    cur, prev = x1, x0

    def step(cur):
        nonlocal prev
        next = cur-f(cur)*(cur-prev)/(f(cur)-f(prev))
        prev = cur
        return next
    i = 0
    while abs(cur-prev) >= eps:
        i += 1
        if logs:
            print(f"i={i}, current={cur}, f(current)={f(cur)}")
        cur = step(cur)
    return cur


def relax(f, x0: float, a: float, b: float, logs: bool=True, eps: float=1e-5, tau: float=0.3)->float:
    input_check(a, b, x0)

    def phi(x):
        return -tau*f(x)+x
    i = 0
    xi = x0
    while abs(phi(xi)-xi) >= eps:
        i += 1
        if logs:
            print(f"i={i}, xi={xi}, f(xi)={f(xi)}")
        xi = phi(xi)
    return phi(xi)


a = 0.0
b = 1.0
x0 = 0.4
x1 = 0.7
print(newton(f, d, x0, a, b))
print(secant(f, x0, x1, a, b))
print(relax(f, x0, a, b))
