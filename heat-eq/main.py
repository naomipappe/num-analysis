import numpy as np


K_C_DIFF_ABS = 273.15
R = 0.15
rho = 790
c = 460
gamma = 140
lambd = 42.1
u_env = 20 + K_C_DIFF_ABS
T = 20*60


alpha = lambd/(c*rho)
T1 = alpha*T/(R**2)
gamma1 = R/lambd * gamma


def u0(x: float) -> float:
    return 100 + 20 * np.sin(x*(R-x)) + K_C_DIFF_ABS


def v0(x: float) -> float:
    return (u0(x)-u_env)/u_env


def to_celcius(temp: float) -> float:
    return temp*u_env+u_env-K_C_DIFF_ABS


N = 10
M = 200
sigm = 0.5
tau, h = T1/M, 1/N
print('Tau =', tau)
print('h =', h)

x = [i*h for i in range(N+1)]

dx = np.zeros_like(x)
dx[0] = (x[1]**3)/(3*h)
dx[N] = (x[N]**3-x[N-1]**3)/(3*h)

for i in range(1, N):
    dx[i] = (x[i+1]**3-x[i-1]**3)/(6*h)

dp = list()

for i in range(1, N+1):
    dp.append((x[i] - h/2)**2)

dp = np.array(dp)
input()
y = [v0(node) for node in x]
print([to_celcius(temp) for temp in y])
for j in range(1, M+1):
    A = np.zeros((N+1, N+1))
    yy = np.zeros_like(x)

    b = sigm*tau/(h**2)*dp[0]
    c = -1/2*dx[0]-b
    phi = -1/2*dx[0]*y[0] - (1 - sigm)*tau/(h**2)*dp[0]*(y[1] - y[0])

    A[0][0] = c
    A[0][1] = b
    yy[0] = phi

    for i in range(1, N):
        d = sigm*tau/(h**2)*dp[i-1]
        b = sigm*tau/(h**2)*dp[i]
        c = -dx[i] - (b + d)
        phi = -1/2*dx[0]*y[0]-(1-sigm)*tau/(h**2) * \
            (dp[i]*(y[i+1]-y[i])-dp[i-1]*(y[i]-y[i-1]))
        A[i][i] = c
        A[i][i-1] = d
        A[i][i+1] = b
        yy[i] = phi
    d = 0
    c - sigm*tau/h*gamma1*x[N]-1/2*dx[N] - d
    phi = (1-sigm)*tau/h*gamma1*x[N]*y[N] #- tau/h*x[N]*gamma1*u_env

    A[N][N-1] = d
    A[N][N] = c
    yy[N] = phi
    y = np.linalg.solve(A, yy)
    print([to_celcius(temp) for temp in y])

print([to_celcius(temp) for temp in y])
