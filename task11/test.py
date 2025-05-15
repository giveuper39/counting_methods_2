import numpy as np
from task11.main import run_test


def test1():
    # Минимизация f(x,y) = (x-1)^2 + (y-2)^2 при условии x^2 + y^2 <= 1
    def f(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2

    x0 = [0.2, 0.2]
    bounds = [(-2, 2), (-2, 2)]
    true_min = 6 - 2 * np.sqrt(5)

    constraints_ineq = [lambda x: 1 - (x[0]**2 + x[1]**2)]

    run_test("Test 1", f, x0, bounds, true_min=true_min, constraints_ineq=constraints_ineq)


def test2():
    # Минимизация f(x,y) = x^2 + y^2 при условиях: x + y >= 1 и x >= 0
    def f(x):
        return x[0]**2 + x[1]**2

    x0 = [0.4, 0.4]
    bounds = [(0, None), (None, None)]
    true_min = 0.5  # При x = 0.5, y = 0.5

    constraints_ineq = [
        lambda x: x[0] + x[1] - 1
    ]

    run_test("Test 2", f, x0, bounds, true_min=true_min, constraints_ineq=constraints_ineq)


def test3():
    # Трехмерная задача: f(x,y,z) = x^2 + y^2 + z^2 при условиях:
    # x + y + z = 1 и x >= 0, y >= 0, z >= 0
    def f(x):
        return x[0]**2 + x[1]**2 + x[2]**2

    x0 = [0.3, 0.3, 0.4]
    bounds = [(0, None)] * 3
    true_min = 1/3  # При x=y=z=1/3

    constraints_eq = [lambda x: x[0] + x[1] + x[2] - 1]

    run_test("Test 3", f, x0, bounds, true_min=true_min, constraints_eq=constraints_eq)


if __name__ == "__main__":
    test1()
    test2()
    test3()