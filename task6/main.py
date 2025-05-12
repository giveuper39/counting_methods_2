import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Equation(ABC):
    @abstractmethod
    def u_true(self, x):
        pass

    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def finite_difference(self, N):
        pass

    def max_error(self, u_num, x):
        return np.max(np.abs(u_num - self.u_true(x)))

    def richardson_error(self, N):
        x1, u1 = self.finite_difference(N)
        x2, u2 = self.finite_difference(2 * N)
        u2_interp = u2[::2]
        R = np.abs(u2_interp - u1) / 3
        return R.max()


def main_loop(eq, N_values, num_draw=20):
    errors = []
    richardson_errors = []

    print(f"\n{'N':>6} | {'max погрешность':>18} | {'погрешность по Ричардсону':>26}")
    print("-" * 55)

    for i, N in enumerate(N_values):
        x, u_num = eq.finite_difference(N)
        err = eq.max_error(u_num, x)
        errors.append(err)

        if i != 0:
            r_err = eq.richardson_error(N)
            richardson_errors.append(r_err)
            print(f"{N:6} | {err:18.6e} | {r_err:26.6e}")
        else:
            print(f"{N:6} | {err:18.6e} | {'-':>26}")

    # График численного и аналитического решения
    plt.figure(figsize=(10, 6))
    x_fine = np.linspace(0, 1, 1000)
    plt.plot(x_fine, eq.u_true(x_fine), label='Аналитическое', color='black')
    x, u_num = eq.finite_difference(num_draw)
    plt.plot(x, u_num, 'o-', label=f'Численное (N={num_draw})')
    plt.title(r"Решение уравнения")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
