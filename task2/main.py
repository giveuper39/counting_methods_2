import numpy as np

from task1.main import get_volume_cond, get_spectral_cond, get_angle_cond, vary_free_coefs


def lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Нулевой элемент на диагонали. LU-разложение невозможно.")
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return L, U


def lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Нулевой элемент на диагонали => система является вырожденной.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    Q = np.zeros((n, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def qr_solve(Q: np.ndarray, R: np.ndarray, b: np.ndarray):
    n = Q.shape[0]
    Qtb = np.dot(Q.T, b)

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if R[i, i] == 0:
            raise ValueError("Нулевой элемент на диагонали => система является вырожденной.")
        x[i] = (Qtb[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x


def print_matrix_and_info(A: np.ndarray, name: str = "A"):
    print(name + ":")
    for i in range(A.shape[0]):
        print("\t ", end=" ")
        for j in range(A.shape[1]):
            print(A[i][j], end=" ")
        print()
    print()

    print("Спектральное число обусловленности:", get_spectral_cond(A))
    print("Объемное число обусловленности:", get_volume_cond(A))
    print("Угловое число обусловленности:", get_angle_cond(A))


def main():
    print("СЛАУ: Ax = b")
    n = int(input("Введите размерность матрицы A (по умолчанию, 3): n = ") or 3)
    print("Введите матрицу A (по строкам):")
    A = np.array([list(map(float, input().split())) for _ in range(n)])
    print("Введите столбец b (в строку): ", end="")
    b = np.array(list(map(float, input().split())))
    if len(b) != n:
        raise ValueError("Неверно введен столбец свободных членов!")

    print_matrix_and_info(A)
    x_exact = np.linalg.solve(A, b)
    print("Точное решение: ", x_exact, "\n")

    L, U = lu_decomposition(A)
    x_lu = lu_solve(L, U, b)
    print_matrix_and_info(L, "L")
    print_matrix_and_info(U, "U")
    print("Решение LU-разложением: ", x_lu, "\n")

    Q, R = qr_decomposition(A)
    x_qr = qr_solve(Q, R, b)
    print_matrix_and_info(Q, "Q")
    print_matrix_and_info(R, "R")
    print("Решение QR-разложением: ", x_qr, "\n")

    b_new = vary_free_coefs(b)
    x_exact_new = np.linalg.solve(A, b_new)
    x_lu_new = lu_solve(L, U, b_new)
    x_qr_new = qr_solve(Q, R, b_new)
    print(x_exact_new, "\n", x_lu_new, "\n", x_qr_new)


if __name__ == '__main__':
    main()
