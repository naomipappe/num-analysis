from typing import Callable, Type

from numpy import linspace, vectorize

from integration_formulas import QuadraticFormula


def _aposteriori_error_estimation(integral_prev: float, integral_next: float, algebraic_precision: int):
    return abs(integral_prev - integral_next) / (2 ** algebraic_precision - 1)


class IntegrationStrategy:
    _description = 'Integration strategy'

    @classmethod
    def calculate(cls, integrand: Callable[[float], float], quadratic_formula: Type[QuadraticFormula],
                  integration_borders: tuple, tolerance: float,
                  integrand_nth_derivative: Callable[[float], float] or None = None) -> tuple:
        raise NotImplementedError('Abstract function called')

    @classmethod
    def description(cls):
        return cls._description


class AprioriEstimationStrategy(IntegrationStrategy):
    _description = 'Apriori error estimation integration strategy'

    @classmethod
    def calculate(cls, integrand: Callable[[float], float], quadratic_formula: Type[QuadraticFormula],
                  integration_borders: tuple, tolerance: float,
                  integrand_nth_derivative: Callable[[float], float] or None = None) -> tuple:
        if integrand_nth_derivative is None:
            raise ValueError('N-th derivative should not be None')

        error, step = cls.apriori_estimation(integration_borders, quadratic_formula,
                                             integrand_nth_derivative, tolerance)
        integral = quadratic_formula.value_composite(integration_borders, integrand, step)

        return integral, step, error

    @classmethod
    def apriori_estimation(cls, integration_borders: tuple, quadratic_formula: Type[QuadraticFormula],
                           integrand_nth_derivative: Callable[[float], float], tolerance: float) -> tuple:
        values = vectorize(integrand_nth_derivative)(linspace(*integration_borders, 10 ** 5))
        dnf_measure = max(abs(values))
        error, step = quadratic_formula.quadratic_formula_step_error(integration_borders, dnf_measure, tolerance)
        return error, step


class RungeStrategy(IntegrationStrategy):
    _description = 'Runge method integration strategy'

    @classmethod
    def calculate(cls, integrand: Callable[[float], float], quadratic_formula: Type[QuadraticFormula],
                  integration_borders: tuple, tolerance: float,
                  integrand_nth_derivative: Callable[[float], float] or None = None) -> tuple:
        algebraic_precision = quadratic_formula.algebraic_precision()
        left_border, right_border = integration_borders
        step = right_border - left_border

        integral_prev = quadratic_formula.value_composite(integration_borders, integrand, step)
        step = step / 2
        integral_next = quadratic_formula.value_composite(integration_borders, integrand, step)

        while _aposteriori_error_estimation(integral_prev, integral_next, algebraic_precision) > tolerance / 2:
            step = step / 2
            integral_prev = integral_next
            integral_next = quadratic_formula.value_composite(integration_borders, integrand, step)

        integral_next = cls._richardson_clarification(integral_next, integral_prev, algebraic_precision)

        return integral_next, step, _aposteriori_error_estimation(integral_prev, integral_next, algebraic_precision)

    @classmethod
    def _richardson_clarification(cls, integral_prev: float, integral_next: float, algebraic_precision: int) -> float:
        return 2 ** algebraic_precision / (2 ** algebraic_precision - 1) \
               * integral_next - 1 / (2 ** algebraic_precision - 1) * integral_prev


class AdaptiveStrategy(IntegrationStrategy):
    _description = 'Adaptive quadratic formula integration strategy'

    @classmethod
    def calculate(cls, integrand: Callable[[float], float], quadratic_formula: Type[QuadraticFormula],
                  integration_borders: tuple, tolerance: float,
                  integrand_nth_derivative: Callable[[float], float] or None = None) -> tuple:
        algebraic_precision = quadratic_formula.algebraic_precision()
        integral = 0
        left_border, right_border = integration_borders
        step = (right_border - left_border) / 10
        adaptive_steps = []
        integral_next, integral_prev = cls.adaptive_iteration(integrand, left_border, quadratic_formula, step)
        while left_border != right_border:
            if (left_border + step) > right_border:
                step = right_border - left_border

            integral_next, integral_prev = cls.adaptive_iteration(integrand, left_border, quadratic_formula, step)

            while _aposteriori_error_estimation(integral_prev, integral_next,
                                                algebraic_precision) > step * tolerance / (right_border - left_border):
                step /= 2
                integral_next, integral_prev = cls.adaptive_iteration(integrand, left_border, quadratic_formula, step)

            adaptive_steps.append(step)
            left_border += step
            integral += integral_next
            step *= 2

        return integral, adaptive_steps, _aposteriori_error_estimation(integral_prev, integral_next,
                                                                       algebraic_precision)

    @classmethod
    def adaptive_iteration(cls, integrand, left_border, quadratic_formula, step):
        integral_prev = quadratic_formula.value_simple((left_border, left_border + step), integrand)
        integral_next = quadratic_formula.value_simple((left_border, left_border + step / 2), integrand) + \
                        quadratic_formula.value_simple((left_border + step / 2, left_border + step), integrand)
        return integral_next, integral_prev
