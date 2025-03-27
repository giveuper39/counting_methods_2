import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from task3.main import main_loop


def generate_matrix(n, density=0.01, dominance_factor=1.5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    num_nonzeros = int(n * (n - 1) * density / 2)
    rows = np.random.choice(n, size=num_nonzeros)
    cols = np.random.choice(n, size=num_nonzeros)
    upper_mask = rows < cols
    rows = rows[upper_mask]
    cols = cols[upper_mask]
    data = np.random.randn(len(rows)) * 0.5
    sparse_upper = coo_matrix((data, (rows, cols)), shape=(n, n))
    matrix = sparse_upper + sparse_upper.T
    diagonal = np.zeros(n)
    for i in range(n):
        row_sum = np.sum(np.abs(matrix[i].data)) - np.abs(matrix[i, i] if matrix.has_canonical_format else 0)
        diagonal[i] = row_sum * dominance_factor + np.random.rand() * 0.1  # small noise
    diag_matrix = sp.diags(diagonal, format='csr')
    result = matrix + diag_matrix
    return result.tocsr()


def generate_b(n, density=0.3, value_range=(-1.0, 1.0), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    b = np.zeros(n)
    num_nonzero = int(n * density)
    if num_nonzero == 0:
        return b

    indices = np.random.choice(n, size=num_nonzero, replace=False)
    min_val, max_val = value_range
    values = np.random.uniform(low=min_val, high=max_val, size=num_nonzero)
    b[indices] = values
    return b


def test_large_1():
    n = 1000
    A = generate_matrix(n, random_seed=1111)
    b = generate_b(n, random_seed=1111)
    main_loop(A.toarray(), b, w=1.1)


def test_diag():
    e = 1e-9
    A = np.array([[16, 5, 0],
                  [5, 100, 4],
                  [0, 4, 30]], float)

    b = np.array([21, 109, 34], float)
    main_loop(A, b, w=1.05, e=e)


def test_large_2():
    n = 10000
    e = 1e-12
    A = generate_matrix(n, random_seed=3333)
    b = generate_b(n, random_seed=3333)
    main_loop(A.toarray(), b, w=1.1, e=e)


def test_positive():
    e = 1e-12
    A = np.array([
        [5, 1, 0, 0, 1],
        [1, 6, 2, 0, 0],
        [0, 2, 7, 1, 0],
        [0, 0, 1, 8, 2],
        [1, 0, 0, 2, 9]
    ], float)
    b = np.array([7, 9, 10, 11, 12], float)
    main_loop(A, b, w=1.1, e=e)