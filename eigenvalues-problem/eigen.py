import numpy as np


def scalar_products_method(matrix, eps=1e-6):
    def mu(x, e):
        return np.dot(np.reshape(x, (n,)), np.reshape(e, (n,)))

    def normalize(x):
        return x/np.linalg.norm(x)

    matrix = np.array(matrix)
    n = len(matrix)
    x0 = np.random.rand(n,1)
    e_cur = np.array(normalize(x0))
    x_cur = np.dot(matrix, e_cur)
    mu_cur = mu(x_cur, e_cur)
    i = 0
    while True:
        i += 1
        e_next = normalize(x_cur)
        x_next = np.dot(matrix, e_next)
        mu_next = mu(x_next, e_next)
        if abs(mu_next-mu_cur) <= eps:
            break
        x_cur = x_next
        mu_cur = mu_next
    lambda_mod_max = mu_next
    return lambda_mod_max


def power_method(matrix, eps=1e-6):
    matrix = np.array(matrix)
    n = len(matrix)
    x0 = np.random.rand(n,1)
    x_cur = np.dot(matrix, x0)
    mu_cur = x_cur[0]/x0[0]
    while True:
        x_next = np.dot(matrix, x_cur)
        mu_next = x_next[0]/x_cur[0]
        if abs(mu_next-mu_cur) <= eps:
            break
        x_cur = x_next
        mu_cur = mu_next
    return mu_next[0]


def jacobi_turn_method(matrix, eps=1e-6):
    def indeces(A):
        result = 0
        row = 0
        col = 0
        for i in range(0, n):
            for j in range(0, n):
                if i == j:
                    continue
                if abs(A[i][j]) > result:
                    result = abs(A[i][j])
                    row = i
                    col = j
        return row, col

    def stop(A):
        result = 0
        for i in range(0, n):
            for k in range(0, n):
                if i == k:
                    continue
                result += A[i][k]**2
        return result

    def turn(A):
        (i, j) = indeces(A)
        w = (A[j][j]-A[i][i])/(2*A[i][j])
        t = -w+np.sign(w)*np.sqrt(1+w**2)
        c = 1/np.sqrt(1+t**2)
        s = t*c
        R = np.eye(n, n)
        R[i][i] = c
        R[i][j] = s
        R[j][i] = -s
        R[j][j] = c
        A = np.matmul(A, R)
        A = np.matmul(np.transpose(R), A)
        return A

    eigen = np.array(matrix)
    n = len(matrix)
    while stop(eigen) >= eps:
        eigen = turn(eigen)

    return sorted(np.diag(eigen), reverse=True)


def max_module_eigenvalue(matrix, eps=1e-6):
    return scalar_products_method(matrix, eps)


def min_module_eigenvalue(matrix, eps=1e-6):
    n = len(matrix)
    lamA = max_module_eigenvalue(matrix)
    C = np.eye(n, n)-np.matmul(matrix, matrix)/lamA**2
    return np.sqrt((1-max_module_eigenvalue(C))*lamA**2)


def min_eigenvalue(matrix, eps=1e-6):
    n = len(matrix)
    B = np.linalg.norm(matrix, np.inf)*np.eye(n, n)-matrix
    lambmin = np.linalg.norm(matrix, np.inf)-max_module_eigenvalue(B)
    return lambmin
