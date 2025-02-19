import numpy as np


def solve_system(A: list[list[float]], b: list[float]) -> list[float]:
    return list(np.linalg.solve(A, b))


def inverse_matrix(A: list[list[float]]) -> list[list[float]]:
    return list(np.linalg.inv(A))


def get_matrix_norm(A: list[list[float]], n: int) -> float:
    return np.sqrt(sum(sum(x ** 2 for x in A[i]) for i in range(n)))


def get_spectral_cond(A: list[list[float]], n: int) -> float:
    return get_matrix_norm(A, n) * get_matrix_norm(inverse_matrix(A), n)


def get_volume_cond(A: list[list[float]], n: int) -> float:
    c = 1
    for i in range(n):
        c *= np.sqrt(sum(x ** 2 for x in A[i]))
    return c / abs(np.linalg.det(A))


def get_angle_cond(A: list[list[float]], n: int) -> float:
    def get_column(mat: list[list[float]], col: int) -> list[float]:
        return [mat[i][col] for i in range(n)]

    def get_row(mat: list[list[float]], row: int) -> list[float]:
        return mat[row]

    def vect_norm(vec: list[float]) -> float:
        return np.sqrt(sum(x**2 for x in vec))

    C = inverse_matrix(A)
    return max(vect_norm(get_row(A, i)) * vect_norm(get_column(C, i)) for i in range(n))



def main():
    print("СЛАУ: Ax = b")
    n = int(input("Введите размерность матрицы A (по умолчанию, 3): n = ") or 3)
    A = [[float(_) for _ in input().split()] for _ in range(n)]
    print("Введите столбец b (в строку): ", end="")
    b = [float(_) for _ in input().split()]
    if len(b) != n:
        raise ValueError("Неверно введен столбец свободных членов!")

    print(get_spectral_cond(A, n), get_volume_cond(A, n), get_angle_cond(A, n))
    x1 = solve_system(A, b)


if __name__ == '__main__':
    main()
