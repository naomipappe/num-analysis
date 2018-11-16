#! usr/bin/env python3
# -*- coding: utf-8 -*-
# x*e^x-x-1; положительный; Ньютона, секущих, релаксации
import math
import numpy as np


def logger(i: int, xi: float, fi: float):
    print(f"i={i}, xi={xi}, f(xi)={fi}")


def fault(x0, a, b):
    return abs(b-x0) if f(x0)*f(b) < 0 else abs(a-x0)


def f(x)->float:
    return math.sin(x+2)-x**2+2*x-1


def d(x)->float:
    return (x+1)*math.e**x-1

def d2(x)->float:
    return  (x+2)*math.e**x


def input_check(a: float, b: float, x0: float):
    if a >= b:
        raise ValueError(f"Invalid boundaries, a={a} >= b ={b}")
    if x0 < a or x0 > b:
        raise ValueError(
            f"Invalid starting approximation, x0={x0} is out of boundaries [{a},{b}]")


def newton(f, d, a: float, b: float, x0: float=0, logs: bool=True, eps: float=1e-6)->float:
    """
    :param f: desired function
    :param d: derrivative of desired function
    :param x0: starting approximation
    :param a: left border
    :param b: right border
    :param logs:True for every iteration output displayed
    :param eps: method accuracy
    :return: approximated root
    """

    if not f(a)*f(b) < 0:
        raise ValueError(f"No root can be found on [{a},{b}]")

    if x0 == 0:
        if f(a)*d2(a) > 0:
            x0 = a
        elif f(b)*d2(b) > 0:
            x0 = b
        else:
            x0 = (a+b)/2

    input_check(a, b, x0)

    x_vals = np.linspace(a, b, min(int(1./eps), 10**5))
    min_val, max_val = np.min(np.abs(d(x_vals))), np.max(np.abs(d2(x_vals)))
    z0 = fault(x0, a, b)
    q = max_val*z0/(2*min_val)

    def step(x):
        return x-f(x)/d(x)

    xi = x0
    i = 0

    while abs(step(xi)-xi) > eps or f(step(xi)) > eps:
        if logs:
            logger(i, xi, f(xi))
        i += 1
        xi = step(xi)

    logger(i, step(xi), f(step(xi)))

    apriori_iter_amount = int(np.ceil(
        np.log2(1+(np.log(eps/(b-a))/np.log(q))))+1)

    print(
        f"Newton method. Apriori iteration number: {apriori_iter_amount},",
        f"aposteriori iteration number: {i}")

    return xi


def secant(f, x0: float, x1: float, a: float, b: float, logs: bool=True, eps: float=1e-6)->float:
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

    while abs(cur-prev) > eps or abs(f(cur)) > eps:
        if logs:
            logger(i, cur, f(cur))

        i += 1
        cur = step(cur)
    logger(i, cur, f(cur))
    return cur


def relax(f, x0: float, a: float, b: float, logs: bool=True, eps: float=1e-6)->float:
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
    min_val, max_val = round(np.min(np.abs(d(x_vals))), 6), round(
        np.max(np.abs(d(x_vals))), 6)
    tau = 2./(min_val+max_val)

    q = round((max_val-min_val)/(min_val+max_val), 6)

    def phi(x)->float:
        return x-np.sign(d(x))*tau*f(x)

    i = 0
    xi = x0

    while abs(phi(xi)-xi) > eps or f(phi(xi))>eps:
        if logs:
            logger(i, xi, f(xi))
        i += 1
        xi = phi(xi)
    logger(i, xi, f(xi))
    z0 = fault(x0, a, b)

    apriori_iter_amount = int(
        np.ceil(np.log(z0 / eps) / np.log(1. / q))) + 1

    print(
        f"Relaxation method. Apriori iteration number: {apriori_iter_amount},",
        f"aposteriori iteration number: {i}")

    return xi


if __name__ == "__main__":
    a, b = 0.0, 0.1
    x0 = 0.0
    x1 = 0.05
   # newton_root = newton(f, d, a, b)
    #print("Newton method root: ", newton_root)
    secant_root = secant(f, x0, x1, a, b)
    print("Secant method root: ", secant_root)
    # relaxation_root = relax(f, x0, a, b)
    # print("Relaxation method root: ", relaxation_root)
