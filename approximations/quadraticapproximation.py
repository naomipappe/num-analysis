import funsys as fs
import numpy as np
from math import pi
from typing import Callable
import matplotlib.pyplot as plt


def dot(a: float, b: float, f: Callable[[float], float],
        phi: Callable[[float], float], n: int = 200) -> float:
    """
    Function for computing dot product of functions using mean rectangles formula
    :param n: Amount of nodes
    :param a: Left border
    :param b: Right border
    :param f: Callable
    :param phi: Callable
    :return: Discrete approximation of dot product of f and phi over [a,b]
    """
    h = (b - a) / n
    nodes = [a + i * h for i in range(n)]
    products = [f(nodes[i] - h / 2) * phi(nodes[i] - h / 2) for i in range(n)]
    return sum(products) * h
    # return integrate.quad(lambda x: f(x)*phi(x),a,b)[0]


def dot_discrete(f: Callable[[float], float],
                 phi: Callable[[float], float], nodes: list) -> float:
    return np.dot(list(map(f, nodes)), list(map(phi, nodes))) / (len(nodes) - 1)


def square_root_method(matrix, b, verbose: bool = False):
    def decompose(decomposed_matrix):
        size = len(decomposed_matrix)
        S = np.zeros((size, size))
        D = np.zeros((size, size))
        D[0][0] = np.sign(decomposed_matrix[0][0])
        S[0][0] = np.sqrt(abs(decomposed_matrix[0][0]))
        for j in range(1, size):
            S[0][j] = decomposed_matrix[0][j] / (S[0][0] * D[0][0])
        for i in range(1, size):
            s = decomposed_matrix[i][i] - \
                sum([D[l][l] * (abs(S[l][i]) ** 2) for l in range(i)])
            D[i][i] = np.sign(s)
            S[i][i] = np.sqrt(abs(s))
            for j in range(i + 1, size):
                S[i][j] = decomposed_matrix[i][j] - \
                          sum([np.conj(S[l][i]) * S[l][j] * D[l][l]
                               for l in range(i)])
                S[i][j] /= (S[i][i] * D[i][i])
        ST = np.transpose(np.conj(S))
        return ST, D, S

    n = len(matrix)
    matrix = np.array(matrix)
    b = np.array(b).reshape((n, 1))
    (ST, D, S) = decompose(matrix)
    C = np.matmul(ST, D)
    X1 = np.zeros(n)
    for i in range(n):
        X1[i] = (b[i] - np.dot(X1, C[i])) / C[i][i]

    X2 = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X2[i] = (X1[i] - np.dot(X2, S[i])) / S[i][i]

    e = b - np.dot(matrix, np.reshape(X2, (n, 1)))
    if verbose:
        print("Невязка системы: ", e)
        print("Норма невязки системы: ", np.linalg.norm(e, np.inf))
    return X2


def function_rescale(f: Callable[[float], float], old_a: float, old_b: float,
                     new_a: float, new_b: float) -> Callable[[float], float]:
    return lambda x: f((old_b - old_a) / (new_b - new_a) * (x - new_a) + old_a)


class QuadraticApproximation:
    def __init__(self, a: float or None = None,
                 b: float or None = None,
                 f: Callable[[float], float] or None = None,
                 fsystem: fs.FunctionSystem or None = None):
        """
        :type a: left border
        :type b: right border
        :type f: function to approximate
        :type f: approximation system of functions
        """
        self.descr = 'Mean quadratic root approximation of function'
        self.a = a
        self.b = b
        self.approx_function = f
        self.fsystem = fsystem
        self.mgaC = None
        self.mgaD = None

    def get_mean_quadratic_approximation(self, n: int = 5):
        if self.mgaC:
            if self.mgaC[1] == n:
                return self.mgaC[0]
        if isinstance(self.fsystem, fs.TrigonometricSystem):
            a0, b0 = 0, 2 * np.pi
            self.approx_function = function_rescale(
                self.approx_function, *self.borders, a0, b0)
        else:
            a0, b0 = self.borders
        matrix, v = self._make_system_continuous(a0, b0, n)
        c = np.linalg.solve(matrix, v)
        self.mgaC = lambda x: np.dot(c, np.array([self.fsystem.get_function(k)(x) for k in range(n+1)])), n
        return self.mgaC[0]

    def _make_system_continuous(self, a0, b0, n):
        v = np.array([dot(a0, b0, self.approx_function, self.fsystem.get_function(i))
                      for i in range(n+1)])
        matrix = np.array([[dot(a0, b0, self.fsystem.get_function(k), self.fsystem.get_function(j))
                            for j in range(n+1)] for k in range(n+1)])
        return matrix, v

    def get_mean_quadratic_approximation_discrete(self, n: int = 5, nodes: list or None = None):
        if self.mgaD is not None:
            if self.mgaD[1] == n:
                return self.mgaD[0]
        if nodes is None:
            if isinstance(self.function_system, fs.TrigonometricSystem):
                a0 = 0
                b0 = 2 * pi
                self.approx_function = function_rescale(self.approx_function, self.a, self.b, a0, b0)
            else:
                a0 = self.a
                b0 = self.b
            nodes = np.linspace(a0, b0, n + 1)
        # TO:DO refactor function rescaling (move rescaling function to FunctionSystem methods)
        # TO:DO refactor this function to make it more readable
        cost_func_prev = float.fromhex('0x1.fffffffffffffp+1023')  # костыль
        cost_func_new = float.fromhex('0x1.ffffffffffffep+1023')  # костыль

        m = 0
        while cost_func_new < cost_func_prev:
            m += 1

            matrix, v = self._make_system_discrete(m, nodes)

            c = np.linalg.solve(matrix, v)

            def result(x):
                return np.dot(c, np.array([self.fsystem.get_function(k)(x) for k in range(m + 1)]))

            def delta(x):
                return result(x) - self.approx_function(x)

            cost_func_prev, cost_func_new = cost_func_new, dot_discrete(delta, delta, nodes) * n / (n - m)
        m -= 1
        matrix, v = self._make_system_discrete(m, nodes)
        c = np.linalg.solve(matrix, v)
        self.mgaD = lambda x: np.dot(c, np.array([self.fsystem.get_function(k)(x) for k in range(m + 1)])), n

        def delta(x):
            return self.mgaD[0](x) - self.approx_function(x)

        cost_func_new = dot_discrete(delta, delta, nodes) * n / (n - m)
        print(f"m = {m}, cost = {cost_func_new}")
        return self.mgaD[0]

    def _make_system_discrete(self, m, nodes):
        v = np.array([
            dot_discrete(self.approx_function, self.fsystem.get_function(i), nodes)
            for i in range(m + 1)])
        matrix = np.array([[dot_discrete(self.fsystem.get_function(j), self.fsystem.get_function(i), nodes)
                            for j in range(m + 1)]
                           for i in range(m + 1)])
        return matrix, v

    def plot_approximation(self, n: int = 5, flag: str = None):
        x = np.linspace(self.a, self.b, 200)
        title = f"{str(self.fsystem)} continuous approximation"
        if flag == 'c':
            title = f"{str(self.fsystem)} continuous approximation"
            mga = self.get_mean_quadratic_approximation(n)
        elif flag == 'd':
            title = f"{str(self.fsystem)} discrete approximation"
            mga = self.get_mean_quadratic_approximation_discrete(n)
        else:
            mga = self.get_mean_quadratic_approximation(n)
        #if isinstance(self.fsystem, fs.TrigonometricSystem):
            #self.approx_function = function_rescale(self.approx_function, 0, 2 * pi, self.a, self.b)
            #mga = function_rescale(mga, 0, 2 * pi, self.a, self.b)
        plt.title(title)
        plt.plot(x, mga(x), color="orange", linestyle='-.', label='Approximation function')
        plt.plot(x, self.approx_function(x), 'k.', label='True function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    @property
    def description(self):
        return self.descr

    @description.setter
    def description(self, new_descr: str):
        self.descr = new_descr

    @property
    def borders(self):
        return self.a, self.b

    @borders.setter
    def borders(self, borders: tuple):
        if borders[0] >= borders[1]:
            self.a = borders[1]
            self.b = borders[0]
        else:
            self.a = borders[0]
            self.b = borders[1]

    @property
    def function_system(self):
        if self.fsystem:
            return self.fsystem.description
        else:
            return "No function system set"

    @function_system.setter
    def function_system(self, new_fs: fs.FunctionSystem):
        self.fsystem = new_fs

