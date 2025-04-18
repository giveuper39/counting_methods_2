import numpy as np

from task4.main import scalar_product_method

def jacobi_rotate(A, p, q):
    A = A.copy()
    n = A.shape[0]

    if A[p, q] == 0:
        return A

    phi = 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])
    c = np.cos(phi)
    s = np.sin(phi)

    for i in range(n):
        if i != p and i != q:
            aip = A[i, p]
            aiq = A[i, q]
            A[i, p] = A[p, i] = c * aip - s * aiq
            A[i, q] = A[q, i] = s * aip + c * aiq

    app = A[p, p]
    aqq = A[q, q]
    apq = A[p, q]

    A[p, p] = c**2 * app - 2 * s * c * apq + s**2 * aqq
    A[q, q] = s**2 * app + 2 * s * c * apq + c**2 * aqq
    A[p, q] = A[q, p] = 0.0

    return A


def jacobi_max(A, tol=1e-10, max_iter=1000):
    A = A.copy()
    n = A.shape[0]
    iterations = 0

    def max_offdiag_indices(A):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j
        return p, q

    for _ in range(max_iter):
        p, q = max_offdiag_indices(A)
        if abs(A[p, q]) < tol:
            break
        iterations += 1
        A = jacobi_rotate(A, p, q)

    eigenvalues = np.diag(A)
    return eigenvalues, iterations


def jacobi_cyclic(A, tol=1e-10, max_iter=1000):
    A = A.copy()
    n = A.shape[0]
    iterations = 0

    for _ in range(max_iter):
        converged = True
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p, q]) > tol:
                    A = jacobi_rotate(A, p, q)
                    iterations += 1
                    converged = False
        if converged:
            break

    eigenvalues = np.diag(A)
    return eigenvalues, iterations


def gershgorin_disks(A):
    disks = []
    n = A.shape[0]
    for i in range(n):
        center = A[i, i]
        radius = sum(abs(A[i, j]) for j in range(n) if j != i)
        disks.append((center, radius))
    return disks

def check_eigen_in_disks(eigenvalues, disks):
    results = []
    for lam in eigenvalues:
        in_disk = any(abs(lam - c) <= r for c, r in disks)
        results.append((lam, in_disk))
    return results



def main_loop(A, eps=1e-6):
    eigen_precise, _ = np.linalg.eig(A)
    eigen_precise = list(map(float, sorted(eigen_precise)))
    print(f"Библиотечная функция\n"
          f"собственные значения: {eigen_precise}")
    circles = gershgorin_disks(A)
    print("=" * 30)
    eps_arr = (eps * 1000, eps, eps / 1000)
    for eps in eps_arr:
        print(f"{eps = }")
        eigen_max, iter_max = jacobi_max(A.copy(), eps)
        eigen_cyclic, iter_cyclic = jacobi_cyclic(A.copy(), eps)

        eigen_max = list(map(float, eigen_max))
        eigen_cyclic = list(map(float, eigen_cyclic))

        print(f"Метод Якоби (максимальный элемент)\n"
              f"собственные значения: {sorted(eigen_max)}\n"
              f"количество итераций: {iter_max}\n")

        print("Метод Якоби (циклический обход)\n"
              f"собственные значения: {sorted(eigen_cyclic)}\n"
              f"количество итераций: {iter_cyclic}")
        print("=" * 30)
    print("Круги Гершгорина:")
    for i, (center, radius) in enumerate(circles):
        print(f"Круг {i + 1}: ({center - radius}, {center + radius})")

    eig_values = sorted(np.concatenate([eigen_max, eigen_cyclic]))
    all_in_circles = all(c[1] for c in check_eigen_in_disks(eig_values, circles))

    print(f"Все собственные значения лежат в объединении кругов: {all_in_circles}")

    max_eigen = scalar_product_method(A, eps)[0]
    print(f"\nПроверка из задания 4: максимальное с.ч. методом скалярных произведений: {float(max_eigen)}\n"
          f"Погрешность с методом Якоби: {abs(float(max_eigen) - max(eig_values))}")


def main():
    A = np.array([
        [4, 1, 0.5],
        [1, 3, 0.2],
        [0.5, 0.2, 2]
    ])
    eps = 1e-9
    main_loop(A, eps)


if __name__ == "__main__":
    main()


