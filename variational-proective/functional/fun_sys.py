from typing import Callable
from sympy import lambdify


class FunctionalSystem:
    def __init__(self, context: dict):
        super().__init__()
        self._context = context

    def get_function(self, k: int):
        raise NotImplementedError


class BasisFunction(FunctionalSystem):
    def __init__(self, context: dict):
        super().__init__(context)
        self._constants = self._context['constants']
        self._k = lambdify('x', self._context['k(x)'](), 'numpy')
        self._borders = self._context['borders']

        self._C = self._borders[1] + self._k(self._borders[1]) *\
            (self._borders[1]-self._borders[0]) / \
            (2*self._k(self._borders[1])+self._constants['alpha_2']
             * (self._borders[1]-self._borders[0]))

        self._D = self._borders[0]-self._k(self._borders[0]) * \
            (self._borders[1]-self._borders[0])/(2*self._k(self._borders[0]) + \
                self._constants['alpha_1']*(self._borders[1]-self._borders[0]))

    def get_function(self, k):
        if k == 1:
            return lambda x: (x-self._C)(x-self._borders[0])**2
        elif k == 2:
            return lambda x: (x-self._D)*(self._borders[1]-x)**2
        else:
            return lambda x: ((self._borders[1]-x)**2)*((x-self._borders[0])**(k-1))

    def A_psi(self) -> float:
        return (self._context['mu_1'](self._borders[1]) - self._constants['alpha_1'] *
                self._context['mu_2'](self._borders[1])/self._constants['alpha_2']) / \
            (-self._k(self._borders[0]) + self._constants['alpha_1']*self._borders[0] -
             (self._constants['alpha_1']/self._constants['alpha_2'])*(self._k(self._borders[1]) +
                                                                      self._constants['alpha_2']*self._borders[1]))
