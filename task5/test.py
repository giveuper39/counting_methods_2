import numpy as np
from task5.main import main_loop


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
        [1e6, 1, 1e6],
        [0, 1e6, 1]
    ], dtype=float)
    eps = 1e-6
    print_matrix(A)
    main_loop(A, eps)


def test_diagonal():
    A = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 5]
    ])
    eps = 1e-10
    print_matrix(A)
    main_loop(A, eps)


def test_symmetric_dense():
    A = np.array([
        [4.0, -30.0, 60.0, -35.0],
        [-30.0, 300.0, -675.0, 420.0],
        [60.0, -675.0, 1620.0, -1050.0],
        [-35.0, 420.0, -1050.0, 700.0]
    ])
    eps = 1e-8
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