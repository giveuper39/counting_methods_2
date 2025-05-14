import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def create_grid(a, T, N, M):
    dx = a / N
    dt = T / M
    x = np.linspace(0, a, N + 1)
    t = np.linspace(0, T, M + 1)
    return x, t, dx, dt


def explicit_scheme(x, t, dx, dt, k, mu_func, mu1_func, mu2_func, f_func=None):
    if f_func is None:
        f_func = lambda x, t: 0.0

    N = len(x) - 1
    M = len(t) - 1
    u = np.zeros((M + 1, N + 1))

    # Начальные условия
    for i in range(N + 1):
        u[0, i] = mu_func(x[i])

    # Граничные условия
    for n in range(M + 1):
        u[n, 0] = mu1_func(t[n])
        u[n, N] = mu2_func(t[n])

    alpha = k * dt / dx ** 2
    print(f"Параметр устойчивости для явной схемы: {alpha:.5f}")
    if alpha > 0.5:
        print("⚠️ Условие устойчивости не выполняется!")
    else:
        print("✅ Условие устойчивости выполняется.")

    for n in range(M):
        for i in range(1, N):
            u[n + 1, i] = u[n, i] + alpha * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) + dt * f_func(x[i], t[n])

    return u


def implicit_scheme(x, t, dx, dt, k, mu_func, mu1_func, mu2_func, f_func=None):
    if f_func is None:
        f_func = lambda x, t: 0.0

    N = len(x) - 1
    M = len(t) - 1
    u = np.zeros((M + 1, N + 1))

    # Начальные условия
    for i in range(N + 1):
        u[0, i] = mu_func(x[i])

    # Граничные условия
    for n in range(M + 1):
        u[n, 0] = mu1_func(t[n])
        u[n, N] = mu2_func(t[n])

    alpha = k * dt / dx ** 2

    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        A[i, i] = 1 + 2 * alpha
        if i > 0:
            A[i, i - 1] = -alpha
        if i < N - 2:
            A[i, i + 1] = -alpha

    for n in range(M):
        b = u[n, 1:N] + dt * f_func(x[1:N], t[n])
        b[0] += alpha * mu1_func(t[n + 1])
        b[-1] += alpha * mu2_func(t[n + 1])
        u[n + 1, 1:N] = np.linalg.solve(A, b)

    return u


def plot_solution(x, t, u, title=""):
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
