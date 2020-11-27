from typing import List, Tuple, Union

import numpy as np
from numanalysis.utilities.util import (decompose, input_check, matrix_norm,
                                        vector_norm)
from numpy.core.defchararray import upper


def square_root_method(
    matrix: np.ndarray or List[List[float or int]],
    vector: np.ndarray or List[float or int],
) -> np.ndarray:

    if isinstance(matrix, list):
        matrix = np.array(matrix)

    n = matrix.shape[0]

    if isinstance(vector, list):
        vector = np.array(vector)

    input_check(matrix, vector)

    s, d = decompose(matrix)

    c = np.dot(s.T, d)
    temp = np.zeros(n)

    for i in range(n):
        temp[i] = (vector[i] - np.dot(temp, c[i])) / c[i][i]

    result = np.zeros(n)
    for i in range(n - 1, -1, -1):
        result[i] = (temp[i] - np.dot(result, s[i])) / s[i][i]
    return result


def jacobi(
    matrix: np.ndarray or List[List[float or int]],
    vector: np.ndarray or List[float or int],
    eps=1e-10,
):
    def step(result_candidate: np.ndarray):
        return -np.dot(A, result_candidate) - np.dot(B, result_candidate) + C

    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if isinstance(vector, list):
        vector = np.array(vector)

    input_check(matrix, vector)

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


def gauss(
    matrix: np.ndarray or List[List[float or int]],
    vector: np.ndarray or List[float or int],
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float, np.ndarray]]:
    input_check(matrix, vector)

    n = matrix.shape[0]
    determinant = 1.0
    identity_matrix = np.identity(n)

    for iteration in range(n):
        # Select the index of max value of the current iteration\column
        max_col_index = np.argmax(np.abs(matrix[iteration:n, [iteration]])) + iteration

        if max_col_index != iteration and verbose:
            determinant *= -1.0

        # Form the matrix permutation
        permutation_matrix = np.identity(n)
        permutation_matrix[[iteration, max_col_index]] = permutation_matrix[
            [max_col_index, iteration]
        ]
        # Permute
        matrix = np.dot(permutation_matrix, matrix)
        vector = np.dot(permutation_matrix, vector)
        identity_matrix = np.dot(permutation_matrix, identity_matrix)

        if verbose:
            determinant *= matrix[iteration, iteration]

        # Calculate the M matrix
        m_matrix = np.identity(n)
        m_matrix[iteration, iteration] = 1.0 / matrix[iteration, iteration]
        for i in range(iteration + 1, n):
            m_matrix[i, iteration] = (
                -matrix[i, iteration] / matrix[iteration, iteration]
            )

        # Form the new iteration matrix
        matrix = np.dot(m_matrix, matrix)
        vector = np.dot(m_matrix, vector)
        identity_matrix = np.dot(m_matrix, identity_matrix)

    result = vector
    for i in range(n - 1, -1, -1):
        result[i] -= sum(matrix[i, j] * result[j] for j in range(i + 1, n))

    if verbose:
        matrix_inverse = np.ndarray(shape=(n, n), dtype=float)
        for iteration in range(n):
            inverse_column = identity_matrix[:, iteration].reshape(-1)
            for i in range(n - 1, -1, -1):
                inverse_column[i] -= sum(
                    matrix[i, j] * inverse_column[j] for j in range(i + 1, n)
                )
                matrix_inverse[:, [iteration]] = inverse_column.reshape(n, 1)

        norm = matrix_norm(matrix)
        inverse_norm = matrix_norm(matrix_inverse)
        condition_number = norm * inverse_norm
        return result, determinant, condition_number, matrix_inverse
    else:
        return result


def seidel(
    matrix: np.ndarray or List[List[float or int]],
    vector: np.ndarray or List[float or int],
    tolerance: float = 1e-4,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    def step(result_candidate: np.ndarray) -> np.ndarray:
        return np.dot(
            np.linalg.inv(diagonal + lower_triangular),
            vector - np.dot(upper_triangular, result_candidate),
        )

    input_check(matrix, vector)

    n = matrix.shape[0]

    lower_triangular = np.tril(matrix, k=-1)
    upper_triangular = np.triu(matrix, k=1)
    diagonal = np.diag(np.diag(matrix))

    current_approximation = np.zeros((n,))
    iteration = 0
    while vector_norm(step(current_approximation) - current_approximation) > tolerance:
        current_approximation = step(current_approximation)
        iteration += 1

    if verbose:
        return current_approximation, iteration
    return current_approximation
