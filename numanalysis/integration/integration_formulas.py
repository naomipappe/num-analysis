from typing import Callable

from numpy import ceil, sqrt


# TODO maybe make them singletones


class QuadraticFormula:
    _algebraic_precision = None
    _description = 'Quadratic formula'

    @classmethod
    def value_simple(cls, integration_borders: tuple, integrand: Callable[[float], float]) -> float:
        raise NotImplementedError('Called to abstract method')

    @classmethod
    def value_composite(cls, integration_borders: tuple, integrand: Callable[[float], float],
                        integration_step: float) -> float:
        raise NotImplementedError('Called to abstract method')

    @classmethod
    def borders_check(cls, left_border, right_border):
        if left_border > right_border:
            raise ValueError(
                'Left integration border should be smaller than right integration border')

    @classmethod
    def quadratic_formula_step_error(cls, integration_borders: tuple, dnf_measure: float, tolerance: float) -> tuple:
        raise NotImplementedError('Called to abstract method')

    @classmethod
    def algebraic_precision(cls):
        return cls._algebraic_precision

    @classmethod
    def description(cls):
        return cls._description


class MeanRectangleFormula(QuadraticFormula):
    _algebraic_precision: int = 2
    _description = 'Mean Rectangle quadratic formula'

    @classmethod
    def value_simple(cls, integration_borders: tuple, integrand: Callable[[float], float]) -> float:
        left_border, right_border = integration_borders
        super().borders_check(left_border, right_border)
        return (right_border - left_border) * integrand((left_border + right_border) / 2)

    @classmethod
    def value_composite(cls, integration_borders: tuple, integrand: Callable[[float], float],
                        integration_step: float) -> float:
        integral = 0
        left_border, right_border = integration_borders
        super().borders_check(left_border, right_border)
        nodes_amount = int(
            ceil((right_border - left_border) / integration_step))

        def node(k: int):
            return left_border + k * integration_step

        for i in range(nodes_amount):
            integral += integrand(node(i) - integration_step / 2)
        return integral * integration_step

    @classmethod
    def quadratic_formula_step_error(cls, integration_borders: tuple, dnf_measure: float, tolerance: float) -> tuple:
        left_border, right_border = integration_borders
        step = sqrt(24 * tolerance / (2 * dnf_measure *
                                      (right_border - left_border)))
        error = (step ** 2) * (right_border - left_border) * dnf_measure / 24
        return error, step


class SimpsonsRule(QuadraticFormula):
    _algebraic_precision: int = 4
    _description = "Simpson's rule quadratic formula"

    @classmethod
    def value_simple(cls, integration_borders: tuple, integrand: Callable[[float], float]) -> float:
        left_border, right_border = integration_borders
        super().borders_check(left_border, right_border)

        return (right_border - left_border) / 6 * (
            integrand(left_border) + 4 * integrand((left_border + right_border) / 2) + integrand(right_border))

    @classmethod
    def value_composite(cls, integration_borders: tuple, integrand: Callable[[float], float],
                        integration_step: float) -> float:
        integral = 0
        left_border, right_border = integration_borders
        super().borders_check(left_border, right_border)
        nodes_amount = int(
            ceil((right_border - left_border) / integration_step))

        def node(k: int) -> float:
            return left_border + k * integration_step

        for i in range(0, nodes_amount, 2):
            integral += integrand(node(i)) + 4 * \
                integrand(node(i + 1)) + integrand(node(i + 2))

        return integration_step / 3 * integral

    @classmethod
    def quadratic_formula_step_error(cls, integration_borders: tuple, dnf_measure: float, tolerance: float) -> tuple:
        left_border, right_border = integration_borders
        step = (tolerance * 2880 / (2 * dnf_measure *
                                    (right_border - left_border))) ** (1 / 4)
        error = dnf_measure * (right_border - left_border) * step ** 4 / 2880
        return error, step


class TrapezoidalFormula(QuadraticFormula):
    _algebraic_precision = 2
    _description = 'Trapezoidal quadratic formula'

    @classmethod
    def value_simple(cls, integration_borders: tuple, integrand: Callable[[float], float]) -> float:
        left_border, right_border = integration_borders
        super().borders_check(left_border, right_border)
        return (right_border - left_border) * (integrand(left_border) + integrand(right_border)) / 2

    @classmethod
    def value_composite(cls, integration_borders: tuple, integrand: Callable[[float], float],
                        integration_step: float) -> float:
        integral = 0
        left_border, right_border = integration_borders
        super().borders_check(left_border, right_border)
        n = int(ceil((right_border - left_border) / integration_step))

        def node(k: int):
            return left_border + k * integration_step

        for i in range(1, n + 1):
            integral += integrand(node(i - 1)) + integrand(node(i))

        return integration_step / 2 * integral

    @classmethod
    def quadratic_formula_step_error(cls, integration_borders: tuple, dnf_measure: float, tolerance: float) -> tuple:
        left_border, right_border = integration_borders
        step = sqrt(12 * tolerance /
                    (2 * (right_border - left_border) * dnf_measure))
        error = (step ** 2) * (right_border - left_border) * dnf_measure / 12
        return error, step
