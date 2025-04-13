import numpy as np
from task4.main import main_loop

def print_matrix(A):
    print()
    for row in A:
        print(" ".join([f"{elem:8.4f}" for elem in row]))
    print()

def test_well_conditioned_small():
    A = np.array([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2]
    ], dtype=float)
    eps = 1e-6
    print_matrix(A)
    main_loop(A, eps)


def test_ill_conditioned():
    A = np.array([
        [1, 1e6, 0],
        [0, 1, 1e6],
        [0, 0, 1]
    ], dtype=float)
    eps = 1e-6
    print_matrix(A)
    main_loop(A, eps)


def test_large_sparse():
    np.random.seed(42)
    n = 100
    density = 0.05
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(max(0, i-3), min(n, i+3)):
            if np.random.rand() < density or i == j:
                A[i,j] = np.random.randn()

    A = (A + A.T)/2
    eps = 1e-6
    main_loop(A, eps)


def test_close_eigenvalues():
    A = np.array([
        [5, 0, 0],
        [0, -4.999, 0],
        [0, 0, 3]
    ], dtype=float)
    eps = 1e-6
    print_matrix(A)
    main_loop(A, eps)


def test_random_diag_dominant():
    np.random.seed(123)
    n = 10
    A = np.random.randn(n, n)
    A = (A + A.T)/2
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1)
    eps = 1e-6
    main_loop(A, eps)