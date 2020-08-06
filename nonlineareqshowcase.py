from src.nonlineareq.non_linear_equations import secant
from numpy import sin, cos


def lhs(x: float) -> float:
    return sin(x+2)-x**2+2*x-1


def lhs_derr(x: float) -> float:
    return -2*x+cos(x+2)+2


def lhs_derr_2(x: float) -> float:
    return -sin(x+2)-2


print(f"Equation root is : {secant(lhs,1,2,1,1.5)}")
