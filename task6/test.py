import numpy as np

from task6.main import main_loop, Equation


class Equation1(Equation):
    """ Уравнение: -u''(x) = pi^2 * sin(pi*x)
        Граничные условия:
            u(0) = 0
            u(1) = 0
        Истинное решение: u(x) = sin(pi*x)
    """

    def u_true(self, x):
        return np.sin(np.pi * x)

    def f(self, x):
        return np.pi ** 2 * np.sin(np.pi * x)

    def finite_difference(self, N):
        h = 1.0 / N
        x = np.linspace(0, 1, N + 1)
        A = np.zeros((N - 1, N - 1))
        b = np.zeros(N - 1)

        for i in range(N - 1):
            A[i, i] = 2
            if i > 0:
                A[i, i - 1] = -1
            if i < N - 2:
                A[i, i + 1] = -1
            b[i] = h ** 2 * self.f(x[i + 1])

        u_inner = np.linalg.solve(A, b)
        u_full = np.zeros(N + 1)
        u_full[1:N] = u_inner
        return x, u_full


class Equation2(Equation):
    """ Уравнение: -u''(x) = 2
        Граничные условия:
            u'(0) = 1 (Неймана слева),
            u(1) = 1 (Дирихле справа)
        Истинное решение: u(x) = -x² + x + 1
    """

    def u_true(self, x):
        return -x ** 2 + x + 1

    def f(self, x):
        return 2

    def finite_difference(self, N):
        h = 1.0 / N
        x = np.linspace(0, 1, N + 1)
        A = np.zeros((N + 1, N + 1))
        b = np.zeros(N + 1)

        for i in range(1, N):
            A[i, i - 1] = 1
            A[i, i] = -2
            A[i, i + 1] = 1
            b[i] = -h ** 2 * self.f(x[i])

        A[0, 0] = -1
        A[0, 1] = 1
        b[0] = h

        A[N, N] = 1
        b[N] = 1

        u = np.linalg.solve(A, b)
        return x, u


N_values = [10, 20, 40, 80, 160, 320, 640]


def test1():
    print("\nЗадача 1: Дирихле")
    eq1 = Equation1()
    main_loop(eq1, N_values, num_draw=40)


def test2():
    print("\nЗадача 2: Смешанные условия")
    eq2 = Equation2()
    main_loop(eq2, N_values, num_draw=60)
