import numpy as np


def solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)


def _inverse_matrix(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(A)


def _get_matrix_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord=2))


def get_spectral_cond(A: np.ndarray) -> float:
    return _get_matrix_norm(A) * _get_matrix_norm(_inverse_matrix(A))


def get_volume_cond(A: np.ndarray) -> float:
    row_norms = np.linalg.norm(A, axis=1)
    c = np.prod(row_norms)
    return c / abs(np.linalg.det(A))


def get_angle_cond(A: np.ndarray) -> float:
    C = _inverse_matrix(A)
    row_norms = np.linalg.norm(A, axis=1)
    col_norms = np.linalg.norm(C, axis=0)
    return np.max(row_norms * col_norms)


def vary_free_coefs(b: np.ndarray) -> np.ndarray:
    from random import randint
    b_new = b.copy()
    for i in range(len(b)):
        b_new[i] += (randint(-100, 100) / 1000)
    return b_new


def main():
    print("СЛАУ: Ax = b")
    n = int(input("Введите размерность матрицы A (по умолчанию, 3): n = ") or 3)
    print("Введите матрицу A (по строкам):")
    A = np.array([list(map(float, input().split())) for _ in range(n)])
    print("Введите столбец b (в строку): ", end="")
    b1 = np.array(list(map(float, input().split())))
    if len(b1) != n:
        raise ValueError("Неверно введен столбец свободных членов!")

    print("Спектральное число обусловленности:", get_spectral_cond(A))
    print("Объемное число обусловленности:", get_volume_cond(A))
    print("Угловое число обусловленности:", get_angle_cond(A))

    x1 = solve_system(A, b1)
    print("Решение системы 1: x =", x1)

    b2 = vary_free_coefs(b1)
    print("b = ", b2)
    x2 = solve_system(A, b2)
    print("Решение системы 2: x =", x2)
    print("Погрешность в решении до и после изменения b: ", abs(np.linalg.norm(x1) - np.linalg.norm(x2)))



if __name__ == '__main__':
    main()
#
# 2
# 1 0.99
# 0.99 0.98
# 1.99 1.97