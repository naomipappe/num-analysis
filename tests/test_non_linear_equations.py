from operator import eq
import unittest
from numpy import sin, cos
from numpy.core.defchararray import equal
from numpy.core.multiarray import result_type
from numanalysis.nonlineareq.non_linear_equations import *


def lhs(x: float) -> float:
    return sin(x+2)-x**2+2*x-1


def lhs_derr(x: float) -> float:
    return -2*x+cos(x+2)+2


def lhs_derr_2(x: float) -> float:
    return -sin(x+2)-2


class TestSecantMethod(unittest.TestCase):
    def test_sine_equation(self):

        # Arrange
        left_border = 0.0
        right_border = 0.1
        initial_root_candidate = 0.0
        next_root_candidate = 0.05
        ROOT = 0.0606335
        equation = NonLinearEquation(
            lhs, left_border, right_border, initial_root_candidate, lhs_derr)

        # Act
        secant_root = Secant.solve(equation)

        # Assert
        self.assertAlmostEqual(
            ROOT, secant_root.calculation_result, delta=1e-6)


class TestNewtonsMethod(unittest.TestCase):
    def test_sine_equation(self):

        # Arrange
        left_border = 0.0
        right_border = 0.1
        initial_root_candidate = 0.0
        ROOT = 0.0606335
        equation = NonLinearEquation(
            lhs, left_border, right_border, initial_root_candidate, lhs_derr, lhs_derr_2)
        # Act
        newtons_root = Newton.solve(equation)

        # Assert
        self.assertAlmostEqual(
            ROOT, newtons_root.calculation_result, delta=1e-6)


class TestRelaxationMethod(unittest.TestCase):
    def test_sine_equation(self):

        # Arrange
        left_border = 0.0
        right_border = 0.1
        initial_root_candidate = 0.0
        ROOT = 0.0606335
        equation = NonLinearEquation(
            lhs, left_border, right_border, initial_root_candidate
        )
        # Act
        relax_root = Relaxation.solve(equation)

        # Assert
        self.assertAlmostEqual(ROOT, relax_root.calculation_result, delta=1e-6)
