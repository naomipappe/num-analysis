import unittest
from numpy import sin, cos
from numanalysis.nonlineareq.non_linear_equations import secant, newton, relax


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

        # Act
        secant_root = secant(lhs, left_border, right_border, initial_root_candidate,
                             next_root_candidate, delta=1e-6)

        # Assert
        self.assertAlmostEqual(ROOT, secant_root, delta=1e-6)


class TestNewtonsMethod(unittest.TestCase):
    def test_sine_equation(self):

        # Arrange
        left_border = 0.0
        right_border = 0.1
        initial_root_candidate = 0.0
        ROOT = 0.0606335

        # Act
        newtons_root = newton(lhs, lhs_derr, lhs_derr_2,
                              left_border, right_border, initial_root_candidate, delta=1e-6)

        # Assert
        self.assertAlmostEqual(ROOT, newtons_root, delta=1e-6)


class TestRelaxationMethod(unittest.TestCase):
    def test_sine_equation(self):

        # Arrange
        left_border = 0.0
        right_border = 0.1
        initial_root_candidate = 0.0
        ROOT = 0.0606335

        # Act
        relax_root = relax(lhs, lhs_derr, left_border,
                           right_border, initial_root_candidate,  delta=1e-6)

        # Assert
        self.assertAlmostEqual(ROOT, relax_root, delta=1e-6)
