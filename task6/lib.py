import numpy as np
import matplotlib.pyplot as plt

# Заполняем матрицу разностными коэффициентами и решаем слау
def solve(a, b, alpha, beta, N, p, q, f):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)

    A = np.zeros((N+1, N+1))
    F = np.zeros(N+1)
    
    A[0, 0] = 1
    F[0] = alpha
    A[N, N] = 1
    F[N] = beta
    
    for i in range(1, N):
        A[i, i - 1] =  1 / h**2 - p(x[i]) / ( 2 * h)
        A[i, i] = -2 / h**2 + q(x[i])
        A[i, i + 1] = 1 / h**2 + p(x[i]) / (2 * h)
        F[i] = f(x[i])
    

    u = np.linalg.solve(A, F)
    return x, u
