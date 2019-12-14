import sympy as sp
from sympy import Symbol, lambdify
from scipy import integrate
import numpy as np


class FiniteElementDiffScheme():
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
            return sum([coeficients[i]*self.__phi(i, x) for i in range(n+1)])
        return approximation

    def __build_system(self, n: int):

        self.__nodes, self.__step = np.linspace(
            self.__a, self.__b, n+1, retstep=True)
        # print(self.__step)
        matrix = np.zeros((n+1, n+1))
        vector = np.zeros((n+1,))

        matrix[0][0] = integrate.quad(lambda x: self.__k(x)/(self.__step**2) + self.__q(x)*(self.__phi(0, x)**2),
                                      self.__nodes[0], self.__nodes[1])[0] + (self.__beta/self.__alpha) * self.__k(self.__a)

        for i in range(1, n):
            matrix[i][i] = integrate.quad(lambda x: self.__k(
                x)/(self.__step**2) + self.__q(x) * (self.__phi(i, x)**2), self.__nodes[i-1], self.__nodes[i+1])[0]

        for i in range(1, n+1):
            matrix[i][i-1] = integrate.quad(lambda x: -self.__k(x)/(self.__step**2) +
                                            self.__q(x) * self.__phi(i, x) * self.__phi(i-1, x), self.__nodes[i-1], self.__nodes[i])[0]

        for i in range(n):
            matrix[i][i+1] = integrate.quad(lambda x: -self.__k(x)/(self.__step**2) +
                                            self.__q(x)*self.__phi(i, x)*self.__phi(i+1, x), self.__nodes[i], self.__nodes[i+1])[0]

        matrix[n][n] = integrate.quad(lambda x: self.__k(x)/(self.__step**2) +
                                      self.__q(x)*(self.__phi(n, x)**2), self.__nodes[n-1], self.__nodes[n])[0] + (self.__delta/self.__gamma) *\
            self.__k(self.__b)

        vector[0] = integrate.quad(lambda x: self.__f(
            x) * self.__phi(0, x), self.__nodes[0], self.__nodes[1])[0]

        for i in range(1, n):
            vector[i] = integrate.quad(lambda x: self.__f(
                x) * self.__phi(i, x), self.__nodes[i-1], self.__nodes[i+1])[0]

        vector[n] = integrate.quad(lambda x: self.__f(
            x)*self.__phi(n, x), self.__nodes[n-1], self.__nodes[n])[0]

        return matrix, vector

    def __phi(self, i: int, x: float):
        if self.__nodes[i] - self.__step <= x <= self.__nodes[i]:
            return (x - (self.__nodes[i] - self.__step)) / self.__step
        elif self.__nodes[i] <= x <= self.__nodes[i] + self.__step:
            return ((self.__nodes[i] + self.__step) - x) / self.__step
        else:
            return 0


class IntegroInterpolationScheme():
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

    def solve(self, nodes: list, step: float):
        matrix, vector = self.__build_system(nodes, step)
        norm_parameter = 2
        print(f'Число обусловенности матрицы по норме <<{norm_parameter}>>', np.linalg.cond(
            matrix, 2))
        results = np.linalg.solve(matrix, vector)
        print(vector - np.dot(matrix, results))
        return results

    def __build_system(self, nodes: list, step: float):
        self.__nodes, self.__step = nodes, step
        n = len(self.__nodes)

        matrix = np.zeros((n, n))
        vector = np.zeros(n)

        # region matrix init
        matrix[0][0] = self.a(1)/self.__step + \
            self.__k(self.__a)*self.__beta/self.__alpha + self.__step/2 * self.d(0)
        matrix[0][1] = -self.a(1)/self.__step

        matrix[n-1][n-2] = -self.a(n-1)/self.__step
        matrix[n-1][n-1] = self.__k(self.__b)*self.__delta/self.__gamma + self.a(n-1)/self.__step +\
            self.__step/2*self.d(n-1)

        for i in range(1, n-1):
            matrix[i][i-1] = -self.a(i)/(self.__step**2)
            matrix[i][i] = (self.a(i) + self.a(i+1))/(self.__step**2) + self.d(i)
            matrix[i][i+1] = -self.a(i+1)/(self.__step**2)
        # endregion
        # region vector init
        vector[0] = (self.__step/2) * self.phi(0)
        vector[n-1] = (self.__step/2) * self.phi(n-1)
        for i in range(1, n-1):
            vector[i] = self.phi(i)
        # endregion
        return matrix, vector

    def d(self, i: int) -> float:
        if i == 0:
            return 2/self.__step * integrate.quad(
                self.__q, self.__nodes[i], self.__nodes[i] + self.__step/2
            )[0]
        elif i == len(self.__nodes) - 1:
            return 2/self.__step * integrate.quad(
                self.__q, self.__nodes[i] - self.__step/2, self.__nodes[i]
            )[0]
        else:
            return 1/self.__step * integrate.quad(
                self.__q, self.__nodes[i] - self.__step /
                2, self.__nodes[i] + self.__step/2
            )[0]

    def a(self, i: int) -> float:
        return 1/self.__step * integrate.quad(
            lambda x: 1/self.__k(x), self.__nodes[i-1], self.__nodes[i]
        )[0]

    def phi(self, i: int):
        if i == 0:
            return 2 / self.__step * integrate.quad(
                self.__f, self.__nodes[i], self.__nodes[i] + self.__step/2
            )[0]
        elif i == len(self.__nodes) - 1:
            return 2 / self.__step * integrate.quad(
                self.__f, self.__nodes[i] - self.__step/2, self.__nodes[i]
            )[0]
        else:
            return 1 / self.__step * integrate.quad(
                self.__f, self.__nodes[i]-self.__step /
                2, self.__nodes[i]+self.__step/2
            )[0]
