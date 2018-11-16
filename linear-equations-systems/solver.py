import numpy as np


def square_root_method(matrix, b):
    def column_sum(matrix, i: int, j: int=0):
        sum_column = 0
        if j == 0:
            for r in range(0, i):
                sum_column += matrix[r, i]*matrix[r, i]
        elif j != 0:
            for r in range(0, i):
                sum_column += matrix[r, i]*matrix[r, j]
        return sum_column
    b = np.reshape(b, (5, 1))
    matrix = np.reshape(matrix, (5, 5))
    S = np.zeros((5, 5))
    #d11 = np.sign(matrix[0, 0])
    S[0, 0] = np.sqrt(matrix[0, 0])
    for i in range(1, 5):
        S[0, i] = matrix[0, i] / S[0, 0]
    for i in range(1, 5):
        S[i, i] = np.sqrt(abs(matrix[i, i]-column_sum(S, i)))
        for j in range(i+1, 5):
            S[i, j] = (matrix[i, j]-column_sum(S, i, j))/S[i, i]
    ST = np.transpose(S)
    y = np.matmul(np.linalg.inv(ST), b)
    print(y)
    x=np.matmul(np.linalg.inv(S),y)
    print(x)

