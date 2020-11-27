from typing import Tuple

from numpy import dot, empty, ndarray
from numpy.linalg.linalg import solve, inv
from numpy.testing import assert_array_almost_equal

from numanalysis.linlag.lineareqsystems import gauss
from numanalysis.utilities.util import vector_norm


def generate_test_system(n: int) -> Tuple[ndarray, ndarray]:
    test_system_matrix: ndarray = empty(shape=(n, n))
    test_system_vector: ndarray = empty(shape=(n,))
    for i in range(n):
        test_system_vector[i] = 1 / (n - i)
        test_system_matrix[:, i] = ((-1) ** (i + 1)) * ((i + 1) + 2)
        test_system_matrix[i, i] = 5 / (i + 1)
    return test_system_matrix, test_system_vector


if __name__ == "__main__":
    n = 2
    matrix, vector = generate_test_system(n)
    solution, determinant, condition_number, inverse_matrix = gauss(
        matrix, vector, True
    )

    print("Linear system of equations solution :")
    print(solution)

    print("Error of the solution : ")
    err = vector - dot(matrix, solution)
    print(err)
    print("Norm of the error : ", vector_norm(err))

    print("Determinant : ", determinant)

    print("Condition number :", determinant)

    print("Inverse : ")
    print(inverse_matrix)

    print("Matrix product with inverse matrix : ")
    print(dot(matrix, inverse_matrix))

    # Check that our solution actually makes sense
    # Solve with numpy
    assert_array_almost_equal(solution, solve(matrix, vector))
    assert_array_almost_equal(dot(matrix, inverse_matrix), dot(matrix, inv(matrix)))
