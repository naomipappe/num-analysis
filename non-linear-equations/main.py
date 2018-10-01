#! usr/bin/env python3
# -*- coding: utf-8 -*-
#x*e^x-x-1; положительный; Ньютона, секущих, релаксации
import math
import numpy as np


def f(x)->float:
    return x*(math.e**x)-x-1


def d(x)->float:
    return math.e**x+x*math.e**x-1


def d2(x)->float:
    return (math.e**x)*(x+2)


def input_check(a: float, b: float, x0: float):
    if a >= b:
        raise ValueError(f"Invalid boundaries, a={a} >= b ={b}")
    if x0 < a or x0 > b:
        raise ValueError(
            f"Invalid starting approximation, x0={x0} is out of boundaries [{a},{b}]")


def newton(f, d, a: float, b: float, x0: float=0, logs: bool=True, eps: float=1e-5)->float:
    """
    :param f: desired function
    :param d: derrivative of desired function
    :param x0: starting approximation
    :param a: left border
    :param b: right border
    :param logs:True for every iteration output displayed
    :param eps: method accuracy
    :return: approxiamted root
    """
    input_check(a, b, x0)
    
    if not f(a)*f(b) < 0:
        raise ValueError(f"No root can be found on [{a},{b}]")
    
    if x0 == 0:
        if f(a)*d2(a)>0:
            x0=a
        elif f(b)*d2(b)>0:
            x0=b

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
    """
    :param f: desired function
    :param x0: first starting appeoximation
    :param x1: second starting approximation
    :param a: left border
    :param b: right border
    :param logs:True for every iteration output displayed
    :param eps: method accuracy
    :return: approxiamted root
    """
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
            print(f"i={i}, xi={cur}, f(xi)={f(cur)}")
        cur = step(cur)
    return cur


def relax(f, x0: float, a: float, b: float, logs: bool=True, eps: float=1e-5)->float:
    """
    :param f: desired function
    :param x0: starting approximation
    :param a: left border
    :param b: right border
    :param logs:True for every iteration output displayed
    :param eps: method accuracy
    :return: approxiamted root
    """
    input_check(a, b, x0)

    x_vals = np.linspace(a, b, min(int(1./eps), 10**5))
    min_val, max_val = np.min(np.abs(d(x_vals))), np.max(np.abs(d(x_vals)))
    tau = 2./(min_val+max_val)
    q = (max_val-min_val)/(min_val+max_val)

    def phi(x)->float:
        return x-np.sign(d(x))*tau*f(x)
    i = 0
    xi = x0
    while abs(phi(xi)-xi) >= eps:
        i += 1
        if logs:
            print(f"i={i}, xi={xi}, f(xi)={f(xi)}")
            xi=phi(xi)

    apriori_iter_amount = int(np.ceil((np.log(eps*(1-q))/(b-a))/np.log(q))+1)

    print(
        f"Relaxation method. Apriori iteration number: {apriori_iter_amount},",
        f"aposteriori iteration number: {i}")

    return phi(xi)


if __name__ == "__main__":
    newton_root = newton(f, d, 0.0, 1.0)
    print("Newton method root: ", newton_root)
    secant_root = secant(f, 0.4, 0.7, 0., 1.)
    print("Secant method root: ", secant_root)
    relaxation_root = relax(f, 0.4, 0.2, 4.0)
    print("Relaxation method root: ", relaxation_root)
