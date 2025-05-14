import numpy as np
from typing import Callable, Tuple

from scipy.interpolate import RegularGridInterpolator


def create_grid(Lx: float, Ly: float, Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Создаёт равномерную сетку."""
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    return np.meshgrid(x, y)


def jacobi(u, f, h, tol=1e-9, max_iter=20000):
    u_new = u.copy()
    Ny, Nx = u.shape

    for _ in range(max_iter):
        diff = 0.0
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                x = j * h
                y = i * h
                u_new[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h**2 * f(x,y))
                diff = max(diff, abs(u_new[i,j] - u[i,j]))

        if diff < tol:
            break

        # Копируем только внутренние точки
        u[1:-1, 1:-1] = u_new[1:-1, 1:-1]

    return u


def solve_elliptic(
        f: Callable[[float, float], float],
        boundary_condition: Callable[[float, float], float],
        Lx: float = 1.0,
        Ly: float = 1.0,
        Nx: int = 10,
        Ny: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Решает эллиптическое уравнение на прямоугольной области с заданными условиями."""
    X, Y = create_grid(Lx, Ly, Nx, Ny)
    h = Lx / (Nx - 1)  # Шаг сетки по x (равный шагу по y для квадратной области)
    u = np.zeros((Ny, Nx))  # Обратите внимание на порядок Ny, Nx

    # Установка граничных условий
    for i in range(Ny):
        for j in range(Nx):
            if i == 0 or i == Ny - 1 or j == 0 or j == Nx - 1:
                u[i, j] = boundary_condition(X[i, j], Y[i, j])

    u = jacobi(u, f, h)
    return X, Y, u


def richardson_error(u_coarse: np.ndarray, u_fine: np.ndarray) -> float:
    """Оценка погрешности по Ричардсону между двумя сетками."""
    nx_coarse, ny_coarse = u_coarse.shape
    nx_fine, ny_fine = u_fine.shape

    # Сетка грубого решения
    x_coarse = np.linspace(0, 1, nx_coarse)
    y_coarse = np.linspace(0, 1, ny_coarse)

    # Сетка точного решения (на которой будем оценивать разницу)
    x_fine = np.linspace(0, 1, nx_fine)
    y_fine = np.linspace(0, 1, ny_fine)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')

    # Создаём интерполятор
    interp = RegularGridInterpolator((x_coarse, y_coarse), u_coarse, method='linear', bounds_error=False, fill_value=None)

    # Точки для интерполяции
    points = np.column_stack([X_fine.ravel(), Y_fine.ravel()])

    # Выполняем интерполяцию
    u_coarse_on_fine = interp(points).reshape(X_fine.shape)

    # Оценка погрешности
    return np.max(np.abs(u_coarse_on_fine - u_fine)) / np.max(np.abs(u_fine))
