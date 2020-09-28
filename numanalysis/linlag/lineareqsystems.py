from typing import List

import numpy as np

from numanalysis.utilities.util import decompose, vector_norm


def square_root_method(matrix: np.ndarray or List[List[float or int]], vector: np.ndarray or List[float]) -> np.ndarray:
    if isinstance(matrix, list):
        matrix = np.array(matrix)

    n = matrix.shape[0]

    if isinstance(vector, list):
        vector = np.array(vector)

    s, d = decompose(matrix)

    c = np.dot(s.T, d)
    temp = np.zeros(n)

    for i in range(n):
        temp[i] = (vector[i] - np.dot(temp, c[i])) / c[i][i]

    result = np.zeros(n)
    for i in range(n - 1, -1, -1):
        result[i] = (temp[i] - np.dot(result, s[i])) / s[i][i]
    return result


def jacobi(matrix : np.ndarray, vector : np.ndarray, eps=1e-10):
    def step(result_candidate : np.ndarray):
        return -np.dot(A, result_candidate) - np.dot(B, result_candidate) + C

    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if isinstance(vector, list):
        vector = np.array(vector)
    
    matrix = np.maximum(matrix, np.transpose(matrix))

    AL = np.tril(matrix, -1)
    AR = np.triu(matrix, 1)
    d_inverse = np.linalg.inv(np.diag(np.diag(matrix)))
    A = np.dot(d_inverse, AL)
    B = np.dot(d_inverse, AR)
    C = np.dot(d_inverse, vector)
    result = np.zeros_like(vector)
    
    while vector_norm(step(result) - result) >= eps:
        result = step(result)
    return result
