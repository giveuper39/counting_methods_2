import numpy as np

from task2.main import main_loop


def test_hilbert_5():
    A = np.array([
        [1, 1/2, 1/3, 1/4, 1/5],
        [1/2, 1/3, 1/4, 1/5, 1/6],
        [1/3, 1/4, 1/5, 1/6, 1/7],
        [1/4, 1/5, 1/6, 1/7, 1/8],
        [1/5, 1/6, 1/7, 1/8, 1/9]
    ])
    b = np.array([1, 1, 1, 1, 1], np.floating)
    main_loop(A, b)


def test_rotate_45_degrees():
    A = np.array([
        [1, 0, 0],
        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    ],
        np.floating)
    b = np.array([1, 2, 3], np.floating)
    main_loop(A, b)


def test_almost_singular_matrix():
    A = np.array([
        [1, 2, 3, 4, 5],
        [2, 4.00001, 6, 8, 10],
        [3, 6, 9.00001, 12, 15],
        [4, 8, 12, 16.00001, 20],
        [5, 10, 15, 20, 25.00001]
    ])
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
