import numpy as np
import matplotlib.pyplot as plt

from lib import *

def plot(f, x_range, y_range, title, xlabel='X', ylabel='Y', zlabel='Z'):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    plt.show(block=True)

def test_sin():
    def f(xy):
        x, y = xy
        return np.sin(x) + np.sin(y)

    def grad(xy):
        x, y = xy
        return np.array([np.cos(x), np.cos(y)])
    
    def hess(xy):
        x, y = xy
        return np.array([
            [-np.sin(x), 0.0],
            [0.0, -np.sin(y)]
        ])

    x0 = np.array([1, 1])
    x_prec = np.array([np.pi / 2, np.pi / 2])

    # При отрицательно определённом гессиане метод Ньютона стремится к лок. максимуму
    print("f(x, y) = sin(x) + sin(y)\n")
    run_all(f, grad, hess, x0, x_prec)
    plot(f, (-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi), 'График функции z = sin(x) + sin(y)')

def test_himmelblau():
    def f(xy):
        x, y = xy
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    def grad(xy):
        x, y = xy
        return np.array([
            4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7),
            2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        ])

    def hess(xy):
        x, y = xy
        return np.array([
            [12 * x**2 + 4 * y - 42, 4 * x + 4 * y],
            [4 * x + 4 * y, 4 * x + 12 * y**2 - 26]
        ])

    x0 = np.array([0, 0])  # Начальная точка
    
    # экстремумы: (-3.78, -3.28), (-2.81, 3.13), (3.58, -1.85), (3, 2), (-0.27, -0.92)
    x_prec = np.array([-3.78, -3.28])
    print("f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2\n")
    run_all(f, grad, hess, x0, x_prec)
    plot(f, (-5, 5), (-5, 5), 'Функция Химмельблау: $f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$')

def test_parabola():
    # x^2 + y^2
    # экстремум - (0, 0)
    def f(xy):
        x, y = xy
        return x**2 + y**2

    # градиент - вектор частных производных
    def grad_f(xy):
        x, y = xy
        return np.array([2 * x, 2 * y])

    # матрица Гессе - частные производные второго порядка по всем переменным
    def hess_f(xy):
        x, y = xy
        return np.array([[2, 0], [0, 4]])

    # начальное прближённое решение
    x0 = np.array([5.0, 5.0])
    x_prec = np.array([0, 0])
    print("f(x, y) = x^2 + y^2\n")
    run_all(f, grad_f, hess_f, x0, x_prec)
    plot(f, (-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi), 'График функции z = x^2 + y^2')

test_himmelblau()
