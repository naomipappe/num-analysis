import unittest
from typing import Tuple, List
from numpy import ndarray
from numpy import array, empty
from numpy.testing import assert_almost_equal
from numpy.linalg import solve
from numanalysis.linlag.solver import square_root_method


def _generate_test_system_ndarray(n: int) -> Tuple[ndarray, ndarray]:
    test_system_matrix: ndarray = empty(shape=(n, n))
    test_system_vector: ndarray = empty(shape=(n, 1))
    for i in range(n):
        test_system_vector[i] = 10*i-n
        for j in range(n):
            if i == j:
                test_system_matrix[i][j] = 10+n+(i+j)/(10+n)
            else:
                test_system_matrix[i][j] = 2*(i+j)/(10+n)
    return test_system_matrix, test_system_vector


def _generate_test_system_list(n: int) -> Tuple[List[List[float]], List[float]]:
    test_system_matrix: List = [[0 for i in range(n)] for j in range(n)]
    test_system_vector: List = [0 for i in range(n)]
    for i in range(n):
        test_system_vector[i] = 10*i-n
        for j in range(n):
            if i == j:
                test_system_matrix[i][j] = 10+n+(i+j)/(10+n)
            else:
                test_system_matrix[i][j] = 2*(i+j)/(10+n)
    return test_system_matrix, test_system_vector


class TestSquareRootMethod(unittest.TestCase):

    def test_system_ndarray_input(self):

        # Arrange
        matrix, vector = _generate_test_system_ndarray(10)

        # Act
        square_root_method_solution = square_root_method(matrix, vector)

        # Assert
        numpy_solution = solve(matrix, vector)
        assert_almost_equal(square_root_method_solution,
                            numpy_solution, decimal=6)

    def test_system_list_input(self):

        # Arrange
        matrix, vector = _generate_test_system_list(10)
        # Act
        square_root_method_solution = square_root_method(matrix, vector)

        # Assert
        numpy_solution: ndarray = solve(
            array(matrix), array(vector)).reshape(10, 1)
        assert_almost_equal(square_root_method_solution,
                            numpy_solution, decimal=6)
