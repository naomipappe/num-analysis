import solver
import numpy as np


def fill_matrix(A):
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                A[i][j] = 10+n+(i+j)/(10+n)
            else:
                A[i][j] = 2*(i+j)/(10+n)


def fill_b(b):
    for i in range(len(b)):
        b[i] = 10*i-n


n = int(input("Введите размерность матрицы: "))
matrix = np.zeros((n, n))
fill_matrix(matrix)
b = np.zeros((n, 1))
fill_b(b)
print("Метод квадратного корня:")
x_s = solver.square_root_method(matrix, b)
print("Решение системы - вектор х, метод квадратного корня")
print(x_s)
print("Метод Якоби:")
x_j = solver.jacobi(matrix, b)
print("Решение системы - вектор х, метод Якоби")
print(x_j)
