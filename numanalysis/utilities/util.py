from numpy import ndarray
from numpy import array, sign, sqrt, conj, transpose, empty, dot, eye,zeros
from typing import List, Type, Tuple


def decompose(matrix: array) -> Tuple[ndarray, ndarray, ndarray]:
    '''
    returns S*DS decomposition of a matrix
    '''
    n = len(matrix)
    s = 0
    S = zeros(shape=(n, n))
    D = zeros(shape=(n, n))
    D[0][0] = sign(matrix[0][0])
    S[0][0] = sqrt(abs(matrix[0][0]))
    
    for j in range(1, n):
        S[0][j] = matrix[0][j]/(S[0][0]*D[0][0])
    for i in range(1, n):
        s = matrix[i][i]-sum([D[l][l]*(abs(S[l][i])**2) for l in range(i)])
        D[i][i] = sign(s)
        S[i][i] = sqrt(abs(s))
        for j in range(i+1, n):
            S[i][j] = matrix[i][j] - \
                sum([conj(S[l][i])*S[l][j]*D[l][l] for l in range(i)])
            S[i][j] /= (S[i][i]*D[i][i])
    ST = transpose(conj(S))
    return (ST, D, S)


def inverse(matrix : ndarray):
    n = matrix.shape[0]
    C = zeros(shape=matrix.shape)
    E = eye(n, n, 0)
    inversed = zeros(matrix.shape)
    ST, D, S = decompose(matrix)
    tmp = dot(ST, D)
    for j in range(0, n):
        C[0][j] = E[0][j]/tmp[0][0]
        for i in range(1, n):
            C[i, j] = E[i, j]-sum([tmp[i, k]*C[k, j]
                                   for k in range(0, i)])
            C[i, j] /= tmp[i, i]

    for j in range(0, n):
        inversed[n-1][j] = C[n-1][j]/S[n-1][n-1]
        for i in range(n-1, -1, -1):
            inversed[i, j] = C[i, j]-sum([S[i, k]*inversed[k, j]
                                     for k in range(i+1, n)])
            inversed[i, j] /= S[i, i]
    return inversed


def vector_norm(x):
    return max(abs(x))


def matrix_norm(matrix):
    max_sum = 0
    for i in range(len(matrix)):
        tmp_sum = sum([abs(matrix[i, j]) for j in range(len(matrix))])
        if tmp_sum > max_sum:
            max_sum = tmp_sum
    return max_sum

def cond(matrix):
    return matrix_norm(inverse(matrix))*matrix_norm(matrix)