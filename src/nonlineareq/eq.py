import numpy as np
from typing import Callable


def fault(initial_root_candidate: float, left_border: float, right_border: float):
    if f(initial_root_candidate)*f(right_border) < 0:
        return abs(right_border-initial_root_candidate)
    else:
        return abs(left_border-initial_root_candidate)


def f(x) -> float:
    return np.sin(x+2)-x**2+2*x-1


def d(x) -> float:
    return (x+1)*np.e**x-1


def d2(x) -> float:
    return (x+2)*np.e**x


def input_check(left_border: float, right_border: float, initial_root_candidate: float):
    if left_border >= right_border:
        raise ValueError(
            f"Invalid boundaries, a={left_border} >= b ={right_border}")
    if initial_root_candidate < left_border or initial_root_candidate > right_border:
        raise ValueError(
            f"Invalid starting approximation, x0={initial_root_candidate} is out of boundaries [{left_border},{right_border}]")


def newton(lhs: Callable[[float], float], lhs_derr: Callable[[float], float],
           left_border: float, right_border: float, initial_root_candidate: float = None, delta: float = 1e-4) -> float:
    """
    lhs:Left-hand side of the equation\n
    lhs_derr: Derrivative of the left - hand side of the equation\n
    initial_root_candidate: Initial root approximation\n
    left_border: Left border of the search interval\n
    right_border: Right border of the search interval\n
    delta: Tolerated method error\n
    returns Approximated root of the lhs in the inteval

    """
    input_check(left_border, right_border, initial_root_candidate)

    if lhs(left_border)*lhs(right_border) >= 0:
        raise ValueError(
            f"No root can be found on [{left_border},{right_border}]")

    if initial_root_candidate is None:
        if lhs(left_border)*d2(left_border) > 0:
            initial_root_candidate = left_border
        elif lhs(right_border)*d2(right_border) > 0:
            initial_root_candidate = right_border
        else:
            initial_root_candidate = (left_border+right_border)/2

    def step(x):
        return x-lhs(x)/lhs_derr(x)

    xi = initial_root_candidate
    i = 0

    while abs(step(xi)-xi) > delta or lhs(step(xi)) > delta:
        i += 1
        xi = step(xi)

    return xi


def secant(f: Callable[[float], float], x0: float, x1: float, a: float, b: float, eps: float = 1e-6) -> float:
    """
    :param f: desired function
    :param x0: first starting appeoximation
    :param x1: second starting approximation
    :param a: left border
    :param b: right border
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
        i += 1
        cur = step(cur)
    return cur


def relax(f: Callable[[float], float], x0: float, a: float, b: float, eps: float = 1e-6) -> float:
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

    def phi(x) -> float:
        return x-np.sign(d(x))*tau*f(x)

    i = 0
    xi = x0

    while abs(phi(xi)-xi) > eps or f(phi(xi)) > eps:
        i += 1
        xi = phi(xi)

    return xi
