import numpy as np
from typing import Callable


def fault(lhs: Callable[[float], float], initial_root_candidate: float, left_border: float, right_border: float):
    if lhs(initial_root_candidate)*lhs(right_border) < 0:
        return abs(right_border-initial_root_candidate)
    else:
        return abs(left_border-initial_root_candidate)


def input_check(left_border: float, right_border: float, initial_root_candidate: float):
    if left_border >= right_border:
        raise ValueError(
            f"Invalid boundaries, a={left_border} >= b ={right_border}")
    if initial_root_candidate < left_border or initial_root_candidate > right_border:
        raise ValueError(
            f"Invalid starting approximation, x0={initial_root_candidate} is out of boundaries [{left_border},{right_border}]")


def newton(lhs: Callable[[float], float], lhs_derr: Callable[[float], float], lhs_derr_2: Callable[[float], float],
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
        if lhs(left_border)*lhs_derr_2(left_border) > 0:
            initial_root_candidate = left_border
        elif lhs(right_border)*lhs_derr_2(right_border) > 0:
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


def secant(lhs: Callable[[float], float], left_border: float, right_border: float,
           initial_root_candidate: float, next_root_candidate: float, delta: float = 1e-6) -> float:
    """
    lhs:Left-hand side of the equation\n
    initial_root_candidate: Initial root approximation\n
    left_border: Left border of the search interval\n
    right_border: Right border of the search interval\n
    delta: Tolerated method error\n
    returns Approximated root of the lhs in the inteval
    """

    input_check(left_border, right_border, initial_root_candidate)
    input_check(left_border, right_border, next_root_candidate)
    if lhs(left_border)*lhs(right_border) >= 0:
        raise ValueError(
            f"No root can be found on [{left_border},{right_border}]")
    cur, prev = next_root_candidate, initial_root_candidate

    def step(cur):
        nonlocal prev
        next = cur-lhs(cur)*(cur-prev)/(lhs(cur)-lhs(prev))
        prev = cur
        return next

    i = 0

    while abs(cur-prev) > delta or abs(lhs(cur)) > delta:
        i += 1
        cur = step(cur)
    return cur


def relax(lhs: Callable[[float], float], lhs_derr: Callable[[float], float], left_border: float, right_border: float,
          initial_root_candidate: float,  delta: float = 1e-6) -> float:
    """
    lhs:Left-hand side of the equation\n
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
    x_vals = np.linspace(left_border, right_border, min(int(1./delta), 10**5))
    min_val, max_val = round(np.min(np.abs(lhs_derr(x_vals))), 6), round(
        np.max(np.abs(lhs_derr(x_vals))), 6)
    tau = 2./(min_val+max_val)

    def phi(x) -> float:
        return x-np.sign(lhs_derr(x))*tau*lhs(x)

    i = 0
    xi = initial_root_candidate

    while abs(phi(xi)-xi) > delta or lhs(phi(xi)) > delta:
        i += 1
        xi = phi(xi)

    return xi
