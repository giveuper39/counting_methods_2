import numpy as np


def iteration_method(A, b, e=1e-6, max_iter=1000):
    x = np.zeros_like(b, dtype=np.float64)
    D = np.diag(A)
    D_inv = 1.0 / D
    R = A - np.diagflat(D)

    for iter in range(max_iter):
        x_new = D_inv * (b - R @ x)
        if np.linalg.norm(x_new - x) < e:
            break
        x = x_new.copy()
    return x, iter + 1


def seidel_method(A, b, e=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)

    for iter in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old) < e:
            break
    return x, iter + 1


def relaxation_method(A, b, w=1.5, e=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)

    if not 0 < w < 2:
        raise ValueError("Параметр релаксации w должен быть в интервале (0, 2)")

    for iter in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (1 - w) * x_old[i] + w * (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old) < e:
            break
    return x, iter + 1


def main_loop(A, b, w=1.5, e=1e-6, max_iter=1000):
    x = np.linalg.solve(A, b)
    x_i, iter_i = iteration_method(A, b, e, max_iter)
    x_s, iter_s = seidel_method(A, b, e, max_iter)
    x_r, iter_r = relaxation_method(A, b, w, e, max_iter)
    print(f"\nТочность: {e = }")
    print(f"Метод итераций: невязка = {np.linalg.norm(x_i - x)}, количество итераций = {iter_i}")
    print(f"Метод Зейделя: невязка = {np.linalg.norm(x_s - x)}, количество итераций = {iter_s}")
    print(f"Метод релаксации ({w = }): невязка = {np.linalg.norm(x_r - x)}, количество итераций = {iter_r}")


def main():
    A = np.array([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2]
    ], dtype=np.float64)

    b = np.array([1, 2, 3], dtype=np.float64)
    main_loop(A, b, w=1.1, e=1e-12)


if __name__ == "__main__":
    main()
