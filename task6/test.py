import numpy as np
import matplotlib.pyplot as plt

from lib import * 

def test_variable_coeffs():
    a, b = 0, 1
    alpha, beta = 1, np.exp(-1)
    N = 30

    def p(x): return x
    def q(x): return 1
    def f(x): return np.exp(-x)

    x, u = solve(a, b, alpha, beta, N, p, q, f)
    exact = np.exp(-x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, u, 'o-', label='Численное решение')
    plt.plot(x, exact, 'r-', label='Точное решение ($e^{-x}$)')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid()
    plt.show()

    error = np.max(np.abs(u - exact))
    print(f"Ошибка: {error:.2e}")


def test_neodnorodnoe():
    a, b = 0, np.pi / 2
    alpha, beta = 0, np.pi
    N = 20

    def p(x): return 0
    def q(x): return 1
    def f(x): return -np.sin(x)

    x, u = solve(a, b, alpha, beta, N, p, q, f)
    exact = np.pi * np.sin(x) + x / 2 * np.cos(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, u, 'o-', label='Численное решение')
    plt.plot(x, exact, 'r-', label='Точное решение')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid()
    plt.show()

    error = np.max(np.abs(u - exact))
    print(f"Ошибка: {error:.2e}")

test_neodnorodnoe()
