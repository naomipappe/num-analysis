import sympy as sp
from sympy import Symbol, lambdify
from numpy import linspace
from math import pow
# from sympy.integrals import integrate
from scipy import integrate
import numpy as np


class DiffScheme():
    def __init__(self, context: dict):
        self.__a, self.__b = context['borders']
        self.__variable = context['variable']
        self.__constants = context['constants']
        self.__nodes, self.__step = None, None
        self.__u_true = context['solution_exact_expr']
        self.__L = context['L']
        self.__f = lambdify(self.__variable, self.__L(
            self.__u_true, self.__variable), 'numpy')
        self.__k = context['k(x)']
        self.__q = context['q(x)']
        self.__alpha = self.__constants['alpha']
        self.__beta = self.__constants['beta']
        self.__delta = self.__constants['delta']
        self.__gamma = self.__constants['gamma']

    def solve(self, n: int):

        matrix, vector = self.__build_system(n)
        coeficients = np.linalg.solve(matrix, vector)

        def approximation(x: float):
            return sum([coeficients[i]*self.phi(i, x) for i in range(n+1)])
        return approximation, self.__nodes

    def __build_system(self, n: int):

        self.__nodes, self.__step = np.linspace(
            self.__a, self.__b, n+1, retstep=True)
        matrix = np.zeros((n+1, n+1))
        vector = np.zeros((n+1,))
        
        matrix[0][0] = integrate.quad(lambda x: self.__k(x)/(self.__step**2) + self.__q(x)*(self.phi(0, x)**2),
                                      self.__nodes[0], self.__nodes[1])[0] + (self.__beta/self.__alpha) * self.__k(self.__a)

        for i in range(1, n):
            matrix[i][i] = integrate.quad(lambda x: self.__k(
                x)/(self.__step**2) + self.__q(x) * (self.phi(i, x)**2), self.__nodes[i-1], self.__nodes[i+1])[0]

        for i in range(1, n+1):
            matrix[i][i-1] = integrate.quad(lambda x: -self.__k(x)/(self.__step**2) +
                                            self.__q(x) * self.phi(i, x) * self.phi(i-1, x), self.__nodes[i-1], self.__nodes[i])[0]

        for i in range(n):
            matrix[i][i+1] = integrate.quad(lambda x: -self.__k(x)/(self.__step**2) +
                                            self.__q(x)*self.phi(i, x)*self.phi(i+1, x), self.__nodes[i], self.__nodes[i+1])[0]

        matrix[n][n] = integrate.quad(lambda x: self.__k(x)/(self.__step**2) +
                                      self.__q(x)*(self.phi(n, x)**2), self.__nodes[n-1], self.__nodes[n])[0] + (self.__delta/self.__gamma) *\
            self.__k(self.__b)

        vector[0] = integrate.quad(lambda x: self.__f(
            x) * self.phi(0, x), self.__nodes[0], self.__nodes[1])[0]

        for i in range(1, n):
            vector[i] = integrate.quad(lambda x: self.__f(
                x) * self.phi(i, x), self.__nodes[i-1], self.__nodes[i+1])[0]

        vector[n] = integrate.quad(lambda x: self.__f(
            x)*self.phi(n, x), self.__nodes[n-1], self.__nodes[n])[0]

        return matrix, vector

    def phi(self, i: int, x: float):
        if self.__nodes[i] - self.__step <= x <= self.__nodes[i]:
            return (x - (self.__nodes[i] - self.__step)) / self.__step
        elif self.__nodes[i] <= x <= self.__nodes[i] + self.__step:
            return ((self.__nodes[i] + self.__step) - x) / self.__step
        else:
            return 0
