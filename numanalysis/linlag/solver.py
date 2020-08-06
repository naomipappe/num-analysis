import numpy as np
from typing import List
from numanalysis.utilities.util import decompose, inverse, matrix_norm, vec_norm, cond
np.set_printoptions(10)


def square_root_method(matrix: np.ndarray or List[List[float or int]], b: np.ndarray or List[int or float]):

    if isinstance(matrix, list):
        matrix = np.array(matrix)
        n = matrix.shape[0]
    else:
        n = len(matrix)
    if isinstance(b, list):
        b = np.array(b).reshape((n, 1))

    ST, D, S = decompose(matrix)

    C = np.dot(ST, D)

    X1 = np.zeros(n)

    for i in range(n):
        X1[i] = (b[i]-np.dot(X1, C[i]))/C[i][i]

    X2 = np.zeros(n)
    for i in range(n-1, -1, -1):
        X2[i] = (X1[i]-np.dot(X2, S[i]))/S[i][i]

    det = np.prod([D[i, i]*S[i, i]**2 for i in range(len(S))])
    print(f"Определитель матрицы А равен = {det}")
    inv_matrix = inverse(matrix)
    inv_matrix = inv_matrix.astype(float)
    print("Обратная матрица матрицы А")
    print(inv_matrix)
    print("cond(A)")
    print(cond(matrix))
    print("Произведение обратной на матрицу системы:")
    print(np.matmul(inv_matrix, matrix))
    X2 = np.reshape(X2, (n, 1))
    e = b-np.dot(matrix, X2)
    print("Вектор невязки решения:")
    print(e)
    print("Норма вектора невязки:")
    print(vec_norm(e))
    return X2


def jacobi(matrix, b, eps=1e-10):
    def step(x):
        return -np.dot(A, x)-np.dot(B, x)+C
    n = len(matrix)
    matrix = np.array(matrix)
    matrix = np.maximum(matrix, np.transpose(matrix))
    b = np.array(b).reshape((n, 1))
    AL = np.tril(matrix, -1)
    AR = np.triu(matrix, 1)
    D_inv = inverse(np.diag(np.diag(matrix)))
    A = np.matmul(D_inv, AL)
    B = np.matmul(D_inv, AR)
    C = np.matmul(D_inv, b)
    xi = np.zeros_like(b)
    i = 0
    while(vec_norm(step(xi)-xi) >= eps):
        i += 1
        xi = step(xi)
    print(f"Количество итераций: {i}")
    print("Невязка решения:")
    print(np.dot(matrix, xi)-b)
    return xi
