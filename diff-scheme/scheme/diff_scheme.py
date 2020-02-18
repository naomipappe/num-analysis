import numpy as np
from scipy import integrate
from sympy import lambdify


class FiniteElementDiffScheme:
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


class IntegroInterpolationScheme:
    def __init__(self, borders: tuple, alpha: float, beta: float, gamma: float, delta: float, mu_1, mu_2, k, p, q):
        self._nodes, self._step = None, None
        self._left_border, self._right_border = borders
        self._k = k
        self._p = p
        self._q = q
        self._mu_1 = mu_1
        self._mu_2 = mu_2
        self._alpha = alpha
        self._beta = beta
        self._delta = delta
        self._gamma = gamma

    def solve(self, n: int, equation_rhs):
        matrix, vector = self.__build_system(n, equation_rhs)
        norm_parameter = 2
        print(f'Число обусловенности матрицы по норме <<{norm_parameter}>>', np.linalg.cond(
            matrix, 2))
        results = np.linalg.solve(matrix, vector)
        print(vector - np.dot(matrix, results))
        return results

    def __build_system(self, n: int, equation_rhs):
        def call_equation_rhs(x: float) -> float:
            return lambdify('x', equation_rhs, 'numpy')(x)

        self._nodes, self._step = np.linspace(self._left_border, self._right_border, n, endpoint=True, retstep=True)
        matrix = np.zeros((n, n))
        vector = np.zeros(n)

        # region matrix init
        matrix[0][0] = self.a(1) / self._step + self._beta + self._step / 2 * self.d(0) - self.s(0) / self._step
        matrix[0][1] = -self.a(1) / self._step + self.s(0) / self._step

        matrix[n - 1][n - 2] = -self.a(n - 1) / self._step - self.s(n - 1) / self._step
        matrix[n - 1][n - 1] = self._delta + self.a(n - 1) / self._step + self._step / 2 * self.d(n - 1) + self.s(
            n - 1) / self._step

        for i in range(1, n - 1):
            matrix[i][i - 1] = -self.a(i) / (self._step ** 2) - self.s(i) / (2 * self._step ** 2)
            matrix[i][i] = (self.a(i) + self.a(i + 1)) / (self._step ** 2) + self.d(i)
            matrix[i][i + 1] = -self.a(i + 1) / (self._step ** 2) + self.s(i) / (2 * self._step ** 2)
            # endregion
        # region vector init

        vector[0] = (self._step / 2) * self.phi(0, call_equation_rhs) + self._mu_1()
        vector[n - 1] = (self._step / 2) * self.phi(n - 1, call_equation_rhs) + self._mu_2()
        for i in range(1, n - 1):
            vector[i] = self.phi(i, call_equation_rhs)

        # endregion
        return matrix, vector

    def d(self, i: int) -> float:
        if i == 0:
            return 2 / self._step * integrate.quad(self._q, self._nodes[i], self._nodes[i] + self._step / 2)[0]
        elif i == len(self._nodes) - 1:
            return 2 / self._step * integrate.quad(self._q, self._nodes[i] - self._step / 2, self._nodes[i])[0]
        else:
            return 1 / self._step * \
                   integrate.quad(self._q, self._nodes[i] - self._step / 2, self._nodes[i] + self._step / 2)[0]

    def a(self, i: int) -> float:
        return (1 / self._step * integrate.quad(lambda x: 1 / self._k(x), self._nodes[i - 1], self._nodes[i])[0]) ** (
            -1)

    def s(self, i: int) -> float:
        if i == 0:
            return integrate.quad(self._p, self._nodes[i], self._nodes[i] + self._step / 2)[0]
        elif i == len(self._nodes) - 1:
            return integrate.quad(self._p, self._nodes[i] - self._step / 2, self._nodes[i])[0]
        else:
            return integrate.quad(self._p, self._nodes[i] - self._step / 2, self._nodes[i] + self._step / 2)[0]

    def phi(self, i: int, equation_rhs):
        if i == 0:
            return 2 / self._step * integrate.quad(equation_rhs, self._nodes[i], self._nodes[i] + self._step / 2)[0]
        elif i == len(self._nodes) - 1:
            return 2 / self._step * integrate.quad(equation_rhs, self._nodes[i] - self._step / 2, self._nodes[i])[0]
        else:
            return 1 / self._step * \
                   integrate.quad(equation_rhs, self._nodes[i] - self._step / 2, self._nodes[i] + self._step / 2)[0]
