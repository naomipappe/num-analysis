import unittest
from numpy import sin, cos
from src.nonlineareq.eq import secant, newton, relax


class TestSecantMethod(unittest.TestCase):
    def test_sine_equation(self):

        # Arrange
        left_border = 0.0
        right_border = 0.1
        initial_root_candidate = 0.0
        next_root_candidate = 0.05
        ROOT = 0.0606335

        def target_equation(x: float) -> float:
            return sin(x+2)-x**2+2*x-1

        # Act
        secant_root = secant(target_equation, initial_root_candidate,
                             next_root_candidate, left_border, right_border, eps=1e-6)

        # Assert
        self.assertAlmostEqual(ROOT, secant_root, delta=1e-6)


class TestNewtonsMethod(unittest.TestCase):
    def test_sine_equation(self):

        # Arrange
        left_border = 0.0
        right_border = 0.1
        initial_root_candidate = 0.0
        ROOT = 0.0606335

        def target_equation(x: float) -> float:
            return sin(x+2)-x**2+2*x-1

        def target_equation_derrivative(x: float) -> float:
            return 2 - 2*x + cos(2 + x)

        # Act
        newtons_root = newton(target_equation, target_equation_derrivative,
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

        def target_equation(x: float) -> float:
            return sin(x+2)-x**2+2*x-1

        # Act
        relax_root = relax(target_equation, initial_root_candidate,
                           left_border, right_border,  eps=1e-6)

        # Assert
        self.assertAlmostEqual(ROOT, relax_root, delta=1e-6)
