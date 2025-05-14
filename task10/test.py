import numpy as np
import matplotlib.pyplot as plt

from task10.main import run_all


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


def test_sin_min():
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

    # Точка, близкая к локальному минимуму (3π/2 ≈ 4.712)
    x0 = np.array([4.5, 4.5])
    x_prec = np.array([3 * np.pi / 2, 3 * np.pi / 2])

    print("f(x, y) = sin(x) + sin(y)\n")
    run_all(f, grad, hess, x0, x_prec)
    plot(f, (-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi), 'График функции z = sin(x) + sin(y)')


def test_himmelblau():
    def f(xy):
        x, y = xy
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def grad(xy):
        x, y = xy
        return np.array([
            4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7),
            2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
        ])

    def hess(xy):
        x, y = xy
        return np.array([
            [12 * x ** 2 + 4 * y - 42, 4 * x + 4 * y],
            [4 * x + 4 * y, 4 * x + 12 * y ** 2 - 26]
        ])

    x0 = np.array([4, 2.5])  # Начальная точка

    # экстремумы: (-3.78, -3.28), (-2.81, 3.13), (3.58, -1.85), (3, 2), (-0.27, -0.92)
    x_prec = np.array([3.0, 2.0]) # Глобальный минимум
    print("f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2\n")
    run_all(f, grad, hess, x0, x_prec)
    plot(f, (-5, 5), (-5, 5), 'Функция Химмельблау: $f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$')


def test_booth():
    # Функция Бута: f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    def f(xy):
        x, y = xy
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def grad_f(xy):
        x, y = xy
        dfdx = 2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5)
        dfdy = 4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)
        return np.array([dfdx, dfdy])

    def hess_f(xy):
        # Гессиан функции Бута
        return np.array([[10, 8],
                         [8, 10]])

    x0 = np.array([-5.0, 5.0])
    x_prec = np.array([1.0, 3.0])  # Точка глобального минимума

    print("Функция Бута: f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2\n")
    run_all(f, grad_f, hess_f, x0, x_prec)
    plot(f, (-10, 10), (-10, 10), 'Функция Бута: $f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2$')


def test_rosenbrock():
    # Функция Розенброка: f(x, y) = (a - x)^2 + b(y - x^2)^2
    a, b = 1, 50

    def f(xy):
        x, y = xy
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    def grad_f(xy):
        x, y = xy
        dfdx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
        dfdy = 2 * b * (y - x ** 2)
        return np.array([dfdx, dfdy])

    def hess_f(xy):
        x, y = xy
        d2fdxx = 2 - 4 * b * (y - 3 * x ** 2)
        d2fdyy = 2 * b
        d2fdxy = -4 * b * x
        return np.array([[d2fdxx, d2fdxy], [d2fdxy, d2fdyy]])

    x0 = np.array([1.2, 1.2])
    x_prec = np.array([1.0, 1.0])  # Глобальный минимум

    print(f"Функция Розенброка: f(x, y) = ({a} - x)^2 + {b}(y - x^2)^2\n")
    run_all(f, grad_f, hess_f, x0, x_prec, learning_rate=0.0001)
    plot(f, (-2, 2), (-1, 3), f'Функция Розенброка: $f(x, y) = ({a} - x)^2 + {b}(y - x^2)^2$')
