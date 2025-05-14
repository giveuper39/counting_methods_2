import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Нужно для 3D графиков

from task9.main import solve_elliptic, richardson_error


def run_test(
        f: callable,
        bc: callable,
        u_exact: callable,
        Lx: float = 1.0,
        Ly: float = 1.0,
        N_list: list = [10, 20, 40],
        title: str = "Тест"
):
    print(f"\n{'-' * 60}")
    print(f"Запуск теста: {title}")
    solutions = []

    for N in N_list:
        X, Y, u = solve_elliptic(f, bc, Lx=Lx, Ly=Ly, Nx=N, Ny=N)
        solutions.append((X, Y, u))

    error_richardson_list = []
    error_exact_list = []

    for i in range(len(solutions) - 1):
        X1, Y1, u1 = solutions[i]
        X2, Y2, u2 = solutions[i + 1]

        # Оценка через Ричардсона
        error_richardson = richardson_error(u1, u2)
        error_richardson_list.append(error_richardson)

        # Вычисление ошибки относительно точного решения
        u_exact_on_grid = u_exact(X2, Y2)  # Используем напрямую сетку
        error_exact = np.max(np.abs(u2 - u_exact_on_grid))
        error_exact_list.append(error_exact)

        print(f"Погрешность Ричардсона ({N_list[i]} → {N_list[i + 1]}): {error_richardson:.2e}")
        print(f"Погрешность к точному решению ({N_list[i + 1]}): {error_exact:.2e}")

    # Визуализация
    X_final, Y_final, u_final = solutions[-1]
    u_exact_final = u_exact(X_final, Y_final)  # Просто применяем функцию к сетке
    plot_comparison(X_final, Y_final, u_final, u_exact_final, title)


def plot_comparison(X, Y, u_num, u_exact, title):
    fig = plt.figure(figsize=(14, 6))

    # Численное решение
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, u_num, cmap='viridis', edgecolor='none')
    ax1.set_title(f"{title}\nЧисленное решение")
    ax1.set_xlabel('x'), ax1.set_ylabel('y'), ax1.set_zlabel('u')

    # Точное решение
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, u_exact, cmap='plasma', edgecolor='none')
    ax2.set_title(f"{title}\nТочное решение")
    ax2.set_xlabel('x'), ax2.set_ylabel('y'), ax2.set_zlabel('u')

    plt.tight_layout()
    plt.show()

    # Разница
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, np.abs(u_num - u_exact), levels=50, cmap='Reds')
    plt.colorbar(label='|u_num - u_exact|')
    plt.title(f"{title} — Разница между численным и точным решением")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# ———————————————————————— ТЕСТЫ ———————————————————————— #

# Тест 1: sin(pi*x)*sin(pi*y), нулевые ГУ
def test_case_1():
    def f(x, y): return -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def bc(x, y): return 0.0

    def u_exact(x, y): return np.sin(np.pi * x) * np.sin(np.pi * y)

    run_test(f, bc, u_exact, title="Тест 1: sin(πx)·sin(πy), u=0 на границе")


# Тест 2: cos(pi*x)*cos(pi*y), частично ненулевые ГУ
def test_case_2():
    def f(x, y):
        return -2 * np.pi ** 2 * np.cos(np.pi * x) * np.cos(np.pi * y)

    def bc(x, y):
        if np.isclose(x, 0):
            return np.cos(np.pi*y)  # u(0,y) = cos(πy)
        elif np.isclose(x, 1):
            return -np.cos(np.pi*y)  # u(1,y) = -cos(πy)
        elif np.isclose(y, 0):
            return np.cos(np.pi*x)  # u(x,0) = cos(πx)
        elif np.isclose(y, 1):
            return -np.cos(np.pi*x)  # u(x,1) = -cos(πx)
        return 0.0

    def u_exact(x, y): return np.cos(np.pi * x) * np.cos(np.pi * y)

    run_test(f, bc, u_exact, title="Тест 2: cos(πx)·cos(πy), частичные ГУ")


# Тест 3: Нелинейный гауссов колокол (5 - e^{-10r^2})
def test_case_3():
    def f(x, y):
        r2 = (x-0.5)**2 + (y-0.5)**2
        return -40 * np.exp(-10 * r2) * (20 * r2 - 2)

    def bc(x, y):
        if np.isclose(x, 0) or np.isclose(x, 1):
            return 5 - np.exp(-10 * (0.25 + (y-0.5)**2))
        elif np.isclose(y, 0) or np.isclose(y, 1):
            return 5 - np.exp(-10 * (0.25 + (x-0.5)**2))
        return 0.0

    def u_exact(x, y):
        return 5 - np.exp(-10 * ((x-0.5)**2 + (y-0.5)**2))

    run_test(
        f,
        bc,
        u_exact,
        Lx=1.0,
        Ly=1.0,
        N_list=[10, 20, 40],
        title="Тест 3: Нелинейный гауссов колокол (5 - e^{-10r^2})"
    )