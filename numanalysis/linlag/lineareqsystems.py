from typing import List

import numpy as np

from numanalysis.utilities.util import decompose, vector_norm


def square_root_method(matrix: np.ndarray or List[List[float or int]], vector: np.ndarray) -> np.ndarray:
    if isinstance(matrix, list):
        matrix = np.array(matrix)

    n = matrix.shape[0]

    if isinstance(vector, list):
        vector = np.array(vector)

    ST, D, S = decompose(matrix)

    C = np.dot(ST, D)


    X1 = np.zeros(n)

    for i in range(n):
        X1[i] = (vector[i]-np.dot(X1, C[i]))/C[i][i]

    X2 = np.zeros(n)
    for i in range(n-1, -1, -1):
        X2[i] = (X1[i]-np.dot(X2, S[i]))/S[i][i]
    return X2


def jacobi(matrix, b, eps=1e-10):
    def step(x):
        return -np.dot(A, x)-np.dot(B, x)+C
    n = len(matrix)
    matrix = np.array(matrix)
    matrix = np.maximum(matrix, np.transpose(matrix))
    b = np.array(b)
    AL = np.tril(matrix, -1)
    AR = np.triu(matrix, 1)
    D_inv = np.linalg.inv(np.diag(np.diag(matrix)))
    A = np.matmul(D_inv, AL)
    B = np.matmul(D_inv, AR)
    C = np.matmul(D_inv, b)
    xi = np.zeros_like(b)
    i = 0
    while(vector_norm(step(xi)-xi) >= eps):
        i += 1
        xi = step(xi)
    return xi
