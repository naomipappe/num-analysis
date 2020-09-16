from numpy import ndarray
from numpy import array, sign, sqrt, conj, transpose, empty, dot, eye,zeros
from numpy.linalg import inv
from typing import List, Type, Tuple

# TODO document this properly
# this is Cholesky decomposition but with tweaks, that is it does not require matrix to be positive - definite
# for instance, it can decompose the next matrix: 
# [1,2]
# [2,1]
# while numpy.linalg.cholseky will fail
def decompose(matrix: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    '''
    returns Cholesky decomposition of a matrix
    '''
    n = matrix.shape[0]
    s = 0
    S = zeros(shape=(n, n))
    D = zeros(shape=(n, n))
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
    return matrix_norm(inv(matrix))*matrix_norm(matrix)