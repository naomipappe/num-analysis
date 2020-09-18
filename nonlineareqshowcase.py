from numpy.testing._private.utils import assert_equal
from numanalysis.nonlineareq.non_linear_equations import *
from numpy import sin, cos


def lhs(x: float) -> float:
    return sin(x+2)-x**2+2*x-1


def lhs_derr(x: float) -> float:
    return -2*x+cos(x+2)+2


def lhs_derr_2(x: float) -> float:
    return -sin(x+2)-2

  
left_border, right_border = 1, 2
initial_root_candidate = 1

equation = NonLinearEquation(lhs, left_border, right_border, 1, lhs_derr)
result = Secant.solve(equation)
print(result)
assert_equal(result.calculation_result, 1.1257726213960026)
