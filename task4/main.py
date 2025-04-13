import numpy as np

def power_method(A, eps=1e-6, max_iter=1000):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    lambda_prev = 0
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        lambda_max = np.dot(v_new, A @ v_new)

        if np.abs(lambda_max - lambda_prev) < eps:
            break

        v = v_new
        lambda_prev = lambda_max

    return lambda_max, v, iterations


def scalar_product_method(A, eps=1e-6, max_iter=1000):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    lambda_prev = 0
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        Av = A @ v
        lambda_max = np.dot(Av, Av) / np.dot(v, Av)
        v_new = Av / np.linalg.norm(Av)

        if np.abs(lambda_max - lambda_prev) < eps:
            break

        v = v_new
        lambda_prev = lambda_max

    return lambda_max, v, iterations


def library_method(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigenvalues))
    lambda_true = eigenvalues[idx]
    v_true = eigenvectors[:, idx]
    return lambda_true, v_true



def main_loop(A, eps=1e-6):
    eps_arr = (eps / 1000, eps, eps * 1000)
    val_prec, vect_prec = library_method(A)
    print(f"Точные значения:\n"
          f"собственное число: {val_prec}\n"
          f"собственный вектор: {vect_prec}")
    for eps in eps_arr:
        val_p, vect_p, iterations_p = power_method(A, eps=eps, max_iter=100000)
        val_s, vect_s, iterations_s = scalar_product_method(A, eps=eps, max_iter=100000)
        print(f"\n{eps = }")
        print(f"Степенной метод:\n"
              f"собственное число: {val_p} (погрешность: {abs(val_p - val_prec)})\n"
              f"собственный вектор: {vect_p}\n"
              f"количество итераций: {iterations_p}\n")

        print(f"Метод скалярных произведений\n"
              f"собственное число: {val_s} (погрешность: {abs(val_s - val_prec)})\n"
              f"собственный вектор: {vect_s}\n"
              f"количество итераций: {iterations_s}")
        print("=" * 20)

def main():
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ], dtype=np.float64)
    eps = 1e-12
    main_loop(A, eps=eps)


if __name__ == "__main__":
    main()
