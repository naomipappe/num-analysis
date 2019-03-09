import funsys as fs
import numpy as np
from math import pi
from math import sin
from typing import Callable
import matplotlib.pyplot as plt
from scipy import integrate


def dot_product(a: float, b: float, f: Callable[[float], float], phi: Callable[[float], float], n:int = 200) -> float:
    """
    Function for computing dot product of functions using mean rectangles formula
    :param n: Amount of nodes
    :param a: Left border
    :param b: Right border
    :param f: Callable
    :param phi: Callable
    :return: Discrete approximation of dot product of f and phi over [a,b]
    """
    # h = (b-a) / n
    # nodes = [a+i*h for i in range(n)]
    # products = [f(nodes[i]-h/2)*phi(nodes[i]-h/2) for i in range(n)]
    # return sum(products)*h
    return integrate.quad(lambda x: f(x)*phi(x),a,b)[0]


def square_root_method(matrix, b):
    def decompose(decomposed_matrix):
        size = len(decomposed_matrix)
        S = np.zeros((size, size))
        D = np.zeros((size, size))
        D[0][0] = np.sign(decomposed_matrix[0][0])
        S[0][0] = np.sqrt(abs(decomposed_matrix[0][0]))
        for j in range(1, size):
            S[0][j] = decomposed_matrix[0][j] / (S[0][0] * D[0][0])
        for i in range(1, size):
            s = decomposed_matrix[i][i] - sum([D[l][l] * (abs(S[l][i]) ** 2) for l in range(i)])
            D[i][i] = np.sign(s)
            S[i][i] = np.sqrt(abs(s))
            for j in range(i + 1, size):
                S[i][j] = decomposed_matrix[i][j] - \
                          sum([np.conj(S[l][i]) * S[l][j] * D[l][l] for l in range(i)])
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

    def get_mean_quadratic_approximation(self, n: int = 20, rescale_f=None):
        if isinstance(self.fsystem, fs.TrigonometricSystem):
            n = (n * 2) + 1
            a0, b0 = -pi, pi
            self.approx_function = function_rescale(self.approx_function, *self.borders, a0, b0)
        else:
            a0, b0 = self.borders
        v = np.array([dot_product(a0, b0, self.approx_function, self.fsystem.get_function(i))
                      for i in range(n)])
        matrix = np.array([[dot_product(a0, b0, self.fsystem.get_function(k), self.fsystem.get_function(j))
                            for j in range(n)] for k in range(n)])
        c = square_root_method(matrix, v)
        return lambda y: np.dot(c, np.array([self.fsystem.get_function(k)(y) for k in range(n)]))

    def plot_approximation(self, n):
        x = np.linspace(-pi, pi, 100)
        mqa = self.get_mean_quadratic_approximation(n)
        # if isinstance(self.fsystem, fs.TrigonometricSystem):
        #     mqa = function_rescale(mqa, -pi, pi, *self.borders)
        plt.plot(x, mqa(x), 'r--', label='Approximation function')
        plt.plot(x, self.approx_function(x), 'b.', label='True function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()

    @property
    def description(self):
        return self.descr

    @description.setter
    def description(self, new_descr: str):
        self.descr = new_descr

    @property
    def borders(self):
        return a, b

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


def f(x):
    return np.sin(x**2)*x


if __name__ == "__main__":
    a = pi/2
    b = 3*pi/2
    obj = QuadraticApproximation(a, b, f, fs.TrigonometricSystem())
    #z = obj.get_mean_quadratic_approximation(5)
    obj.plot_approximation(5)
