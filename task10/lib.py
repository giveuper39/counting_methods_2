import numpy as np
from scipy.optimize import line_search

def print_res(x_calc, x_prec, iter):
    print(f"Вычисленный экстремум: {x_calc}")
    print(f"Ошибка: {np.linalg.norm(x_calc - x_prec)}")
    print(f"Количество итераций: {iter}")
    print()
    print("=" * 20)
    print()


def run_all(f, grad_f, hess_f, x0, x_prec):
    x_gd, _, iter_gd = gradient_descent(f, grad_f, x0)
    x_sd, _, iter_sd = fast_descent(f, grad_f, x0)
    x_hb, _, iter_hb = heavy_ball(f, grad_f, x0)
    x_nt, _, iter_nt = newton(f, grad_f, hess_f, x0)

    print("Реализация трёх методов первого порядка")
    print()
    print("Градиентный спуск:")
    print_res(x_gd, x_prec, iter_gd)

    print("Быстрый спуск:")
    print_res(x_sd, x_prec, iter_sd)

    print("Метод тяжелого шарика:")
    print_res(x_hb, x_prec, iter_hb)

    print("Реализация Метода Ньютона первого порядка:")
    print_res(x_nt, x_prec, iter_nt)

def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    x = x0.copy()
    history = [x0]
    iter = 0
    for i in range(max_iter):
        grad = grad_f(x)
        x_new = x - learning_rate * grad
        history.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            iter = i
            break
        x = x_new
    return x, history, iter


def fast_descent(f, grad_f, x0, max_iter=1000, tol=1e-6):
    x = x0.copy()
    history = [x0]
    iter = 0
    for i in range(max_iter):
        grad = grad_f(x)
        alpha = line_search(f, grad_f, x, -grad)[0]
        if alpha is None:
            alpha = 0.01
        x_new = x - alpha * grad
        history.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            iter = i
            break
        x = x_new
    return x, history, iter

def heavy_ball(f, grad_f, x0, learning_rate=0.01, momentum=0.9, max_iter=1000, tol=1e-6):
    x = x0.copy()
    history = [x0]
    v = np.zeros_like(x0)
    iter = 0
    for i in range(max_iter):
        grad = grad_f(x)
        v = momentum * v + learning_rate * grad
        x_new = x - v
        history.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            iter = i
            break
        x = x_new
    return x, history, iter

def newton(f, grad_f, hess_f, x0, max_iter=1000, tol=1e-6):
    x = x0.copy()
    history = [x0]
    iter = 0
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        delta_x = np.linalg.solve(hess, grad)
        x_new = x - delta_x
        history.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            iter = i
            break
        x = x_new
    return x, history, iter
