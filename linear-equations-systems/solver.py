import numpy as np
np.set_printoptions(10)

def decompose(A):
    n = len(A)
    s = 0
    S = np.zeros((n, n))
    D = np.zeros((n, n))
    D[0][0] = np.sign(A[0][0])
    S[0][0] = np.sqrt(abs(A[0][0]))
    for j in range(1, n):
        S[0][j] = A[0][j]/(S[0][0]*D[0][0])
    for i in range(1, n):
        s = A[i][i]-sum([D[l][l]*(abs(S[l][i])**2) for l in range(i)])
        D[i][i] = np.sign(s)
        S[i][i] = np.sqrt(abs(s))
        for j in range(i+1, n):
            S[i][j] = A[i][j] - \
                sum([np.conj(S[l][i])*S[l][j]*D[l][l] for l in range(i)])
            S[i][j] /= (S[i][i]*D[i][i])
    ST = np.transpose(np.conj(S))
    return (ST, D, S)


def inverse(A):
    n = len(A)
    C = np.zeros((n, n))
    E = np.eye(n, n, 0)
    REV = np.zeros((n, n))
    ST, D, S = decompose(A)
    tmp = np.matmul(ST, D)
    for j in range(0, n):
        C[0][j] = E[0][j]/tmp[0][0]
        for i in range(1, n):
            C[i, j] = E[i, j]-sum([tmp[i, k]*C[k, j]
                                   for k in range(0, i)])
            C[i, j] /= tmp[i, i]

    for j in range(0, n):
        REV[n-1][j] = C[n-1][j]/S[n-1][n-1]
        for i in range(n-1, -1, -1):
            REV[i, j] = C[i, j]-sum([S[i, k]*REV[k, j]
                                     for k in range(i+1, n)])
            REV[i, j] /= S[i, i]
    return REV


def square_root_method(matrix, b):
    def cond(matrix):
        return np.linalg.norm(inverse(matrix),2)*np.linalg.norm(matrix,2)
    n = len(matrix)
    matrix = np.array(matrix)
    b = np.array(b).reshape((n, 1))
    b = b.astype(float)
    matrix = matrix.astype(float)
    (ST, D, S) = decompose(matrix)
    C = np.matmul(ST, D)
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
    print(np.matmul(inv_matrix,matrix))
    X2 = np.reshape(X2, (n, 1))
    e=b-np.dot(matrix,X2)
    print("Вектор невязки решения:")
    print(e)
    print("Норма вектора невязки:")
    print(np.linalg.norm(e,np.inf))
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
    while(np.linalg.norm(step(xi)-xi, np.inf) >= eps):
        i += 1
        xi = step(xi)
    print(f"Количество итераций: {i}")
    print("Невязка решения:")
    print(np.dot(matrix, xi)-b)
    return xi
