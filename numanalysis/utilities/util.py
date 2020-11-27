from typing import List, Tuple

from numpy import ndarray, sign, sqrt, zeros, isclose
from numpy.linalg import inv, det


def decompose(matrix: ndarray) -> Tuple[ndarray, ndarray]:
    """

    Summary
    -------
    This is Cholesky decomposition but with tweaks, that is it does not require matrix to be positive - definite.

    Parameters
    ----------
    :param matrix: ndarray Matrix to find a decomposition for

    Returns
    -------
    Tuple[ndarray, ndarray] : Choletsky decomposition of a matrix

    """

    n = matrix.shape[0]
    s, d = zeros(shape=(n, n)), zeros(shape=(n, n))

    for i in range(n):
        temp = matrix[i][i] - sum(s[k, i] ** 2 * d[k][k] for k in range(i))

        d[i, i], s[i, i] = sign(temp), sqrt(abs(temp))
        s[i, i + 1 : n] = matrix[i, i + 1 : n] - sum(
            s[k, i] * d[k, k] * s[k, i + 1 : n] for k in range(i)
        )

        s[i, i + 1 : n] /= s[i, i] * d[i, i]

    return s, d


def vector_norm(x: ndarray) -> float:
    return max(abs(x))


def matrix_norm(matrix: ndarray) -> float:
    max_sum = 0
    for i in range(len(matrix)):
        tmp_sum = sum([abs(matrix[i, j]) for j in range(len(matrix))])
        if tmp_sum > max_sum:
            max_sum = tmp_sum
    return max_sum


def cond(matrix: ndarray) -> float:
    return matrix_norm(inv(matrix)) * matrix_norm(matrix)


def input_check(
    matrix: ndarray or List[List[float or int]], vector: ndarray or List[float or int]
):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix should be square")
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[0]:
        raise ValueError("Vector should be of vector type and appropriate dimension")
    if isclose(det(matrix),0):
        raise ValueError('Matrix should have non-zero determinant')