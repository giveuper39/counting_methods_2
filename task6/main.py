import numpy as np
import matplotlib.pyplot as plt

from lib import *

# u'' + u = 0, u(0) = 0, u(3 * pi / 2) = -1 (решение: sin(x))
a, b = 0, 3 * np.pi / 2
alpha, beta = 0, -1
N = 10

# Многочлены перед каждой производной
def p(x): return 0
def q(x): return 1
def f(x): return 0

x, u = solve(a, b, alpha, beta, N, p, q, f)


exact = np.sin(x)


plt.plot(x, u, 'o-', label='Численное решение')
plt.plot(x, exact, 'r-', label='Точное решение')
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Решение краевой задачи')
plt.grid()
plt.show(block=True)


error = np.max(np.abs(u - exact))
print(f"Шаг h = {(b - a) / N:.4f}, ошибка = {error:.4f}", end='\n\n')




def richardson_extrapolation(a, b, alpha, beta, p, q, f, N_list):
    errors = []
    h_list = []
    for N in N_list:
        h = (b - a) / N
        x, u = solve(a, b, alpha, beta, N, p, q, f)
        exact = np.sin(x)
        error = np.max(np.abs(u - exact))
        print(f"error (max|u - exact|): {error:.5f}, {h = :.4f}")
        errors.append(error)
        h_list.append(h)
    return h_list, errors

N_list = [10, 20, 40, 80, 160]
h_list, errors = richardson_extrapolation(a, b, alpha, beta, p, q, f, N_list)
