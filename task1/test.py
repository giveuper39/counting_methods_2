import numpy as np

from task1.main import solve_system, get_angle_cond, get_spectral_cond, get_volume_cond, vary_free_coefs


def print_matrix(A: np.ndarray):
    for i in range(A.shape[0]):
        print("\t ", end=" ")
        for j in range(A.shape[1]):
            print(A[i][j], end=" ")
        print()
    print()


def main_loop(A: np.ndarray, b: np.ndarray):
    print("A = ")
    print_matrix(A)
    print("b = ", b)
    x = solve_system(A, b)
    print("x = ", x)
    b_new = vary_free_coefs(b)
    print("b = ", b_new)
    x_new = solve_system(A, b_new)
    print("x = ", x_new)
    print("Спектральное число обусловленности:", get_spectral_cond(A))
    print("Объемное число обусловленности:", get_volume_cond(A))
    print("Угловое число обусловленности:", get_angle_cond(A))
    error = abs(np.linalg.norm(x) - np.linalg.norm(x_new))
    print("Погрешность: ", error)


def test_rotation_45_degrees():
    A = np.array([
        [1, 0, 0],
        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    ],
        np.floating)
    b = np.array([1, 2, 3], np.floating)
    main_loop(A, b)


def test_hilbert_matrix_4():
    n = 4
    A = np.array([
        [1, 1 / 2, 1 / 3, 1 / 4],
        [1 / 2, 1 / 3, 1 / 4, 1 / 5],
        [1 / 3, 1 / 4, 1 / 5, 1 / 6],
        [1 / 4, 1 / 5, 1 / 6, 1 / 7]
    ])
    b = np.array([1, 2, 3, 4], np.floating)
    main_loop(A, b)


def test_almost_singular_matrix():
    n = 4
    A = np.array([
        [1, 2, 3, 4],
        [2, 4.00001, 6, 8],
        [3, 6, 9.00001, 12],
        [4, 8, 12, 16.00001]
    ])
    b = np.array([1, 2, 3, 4], np.floating)
    main_loop(A, b)


def test_large_difference_matrix():
    n = 5
    A = np.array([
        [1, 1e3, 1e6, 1e9, 1e12],
        [1e-3, 2, 1e3, 1e6, 1e9],
        [1e-6, 1e-3, 3, 1e3, 1e6],
        [1e-9, 1e-6, 1e-3, 4, 1e3],
        [1e-12, 1e-9, 1e-6, 1e-3, 5]
    ], np.floating)
    b = np.array([1, 2, 3, 4, 5], np.floating)
    main_loop(A, b)


def test_tridiagonal_matrix():
    A = np.array([
        [4, 1, 0, 0, 0],
        [1, 4, 1, 0, 0],
        [0, 1, 4, 1, 0],
        [0, 0, 1, 4, 1],
        [0, 0, 0, 1, 4]
    ], np.floating)
    b = np.array([1, 2, 3, 4, 5], np.floating)
    main_loop(A, b)