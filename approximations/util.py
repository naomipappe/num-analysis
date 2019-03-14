from typing import Callable

import numpy as np


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
    return np.dot(list(map(f, nodes)), list(map(phi, nodes))) / (len(nodes))


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