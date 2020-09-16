from numpy import empty, ndarray
from typing import Tuple
from numanalysis.linlag.linalgmethods import square_root_method


def _generate_test_system(n: int) -> Tuple[ndarray, ndarray]:
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

square_root_method(*_generate_test_system(5))

