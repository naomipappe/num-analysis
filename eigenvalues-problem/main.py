import eigen
import numpy as np

def fill_matrix(A):
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                A[i][j] = 10+n+(i+j)/(10+n)
            else:
                A[i][j] = 2*(i+j)/(10+n)


n = int(input("Введите размерность матрицы: "))
matrix = np.zeros((n, n))
fill_matrix(matrix)
eps=1e-5
lambda_max_mod = eigen.max_module_eigenvalue(matrix,eps)
lambda_min_mod = eigen.min_module_eigenvalue(matrix,eps)
lambda_min = eigen.min_eigenvalue(matrix,eps)
lambdas = eigen.jacobi_turn_method(matrix,eps)
print("Максимальное по модулю собственное число:", lambda_max_mod)
print("Минимальное по модулю собственное число:", lambda_min_mod)
print("Минимальное собственное число:", lambda_min)
print("Собственные значения матрицы:")
print(lambdas)
