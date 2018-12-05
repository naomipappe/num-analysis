import solver
import numpy as np

matrix_square = [[1, 3, -2, 0, -2],
                 [3, 4, -5, 1, -3],
                 [-2, -5, 3, -2, 2],
                 [0, 1, -2, 5, 3],
                 [-2, -3, 2, 3, 4]]
matrix_jacobi = [[11, 3, -2, 0, -2],
                 [3, 14, -5, 1, -3],
                 [-2, -5, 13, -2, 2],
                 [0, 1, -2, 15, 3],
                 [-2, -3, 2, 3, 14]]
b=[0.5,5.4,5.0,7.5,3.3]
print("Метод квадратного корня:")
x_s = solver.square_root_method(matrix_square, b)
print("Решение системы - вектор х, метод квадратного корня")
print(x_s)
print("Метод Якоби:")
x_j = solver.jacobi(matrix_jacobi, b)
print("Решение системы - вектор х, метод Якоби")
print(x_j)

