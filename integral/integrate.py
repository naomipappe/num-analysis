from typing import Callable, Type

from integration_formulas import QuadraticFormula
from integration_strategy import IntegrationStrategy


class Integral:
    def __init__(self):
        self._description = 'Integral of function'
        self.borders = None, None
        self.integrand = None
        self.integrand_nth_derivative = None

    def integrate(self, tolerance: float, strategy: Type[IntegrationStrategy], formula: Type[QuadraticFormula]):
        integral, step, error = strategy.calculate(self.integrand, formula, self.borders, tolerance,
                                                   self.integrand_nth_derivative)
        print()
        print(formula.description())
        print(strategy.description())
        print(f"Integral of functions over interval [{self.borders[0]},{self.borders[1]}]")
        print(f'I = {integral}')
        if isinstance(step, list):
            steps_info = ' '.join(map(str, step))
            min_step = min(step)
            print('Integration steps:', steps_info)
            print('Minimal step:', min_step)
        else:
            print('Integration step:', step)
        print('Error:', error)
        return integral

    @property
    def borders(self) -> tuple:
        return self._left_border, self._right_border

    @borders.setter
    def borders(self, integration_border: tuple):
        self._left_border, self._right_border = integration_border

    @property
    def integrand(self) -> Callable[[float], float]:
        return self._integrand

    @integrand.setter
    def integrand(self, integrand: Callable[[float], float]):
        self._integrand = integrand

    @property
    def integrand_nth_derivative(self) -> Callable[[float], float]:
        return self._dnf

    @integrand_nth_derivative.setter
    def integrand_nth_derivative(self, integrand_nth_derivative: Callable[[float], float]):
        self._dnf = integrand_nth_derivative
