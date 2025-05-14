from task8.main import explicit_scheme, implicit_scheme, create_grid, plot_solution

import numpy as np


def run_test(params, grid_params_list):
    a = params['a']
    T = params['T']
    k = params['k']
    mu = params['mu']
    mu1 = params['mu1']
    mu2 = params['mu2']
    title_mu1 = params['title_mu1']
    title_mu2 = params['title_mu2']
    title_case = params['title_case']

    print("\n" + "=" * 50)
    print(f"ТЕСТ: {title_case}")
    print(f"Граничные условия: {title_mu1}, {title_mu2}")

    print("=== Явная схема ===")
    for grid in grid_params_list:
        x, t, dx, dt = create_grid(a=a, T=T, **grid)
        u_explicit = explicit_scheme(x, t, dx, dt, k=k, mu_func=mu, mu1_func=mu1, mu2_func=mu2)
        stability = 'нарушена' if k * dt / dx ** 2 > 0.5 else 'соблюдена'
        plot_title = f"{title_case}: Явная схема ({stability})"
        plot_solution(x, t, u_explicit, title=plot_title)

    print("\n=== Неявная схема ===")
    x, t, dx, dt = create_grid(a=a, T=T, N=50, M=100)
    u_implicit = implicit_scheme(x, t, dx, dt, k=k, mu_func=mu, mu1_func=mu1, mu2_func=mu2)
    plot_title = f"{title_case}: Неявная схема"
    plot_solution(x, t, u_implicit, title=plot_title)


def test_case_1():
    params = {
        'a': 1.0,
        'T': 1.0,
        'k': 0.1,
        'mu': lambda x: np.sin(np.pi * x),
        'mu1': lambda t: 0.0,
        'mu2': lambda t: 0.0,
        'title_mu1': 'u(0,t) = 0',
        'title_mu2': 'u(1,t) = 0',
        'title_case': 'Стационарный случай'
    }

    grid_params_list = [
        {"N": 50, "M": 100},  # сильно нарушено
        {"N": 50, "M": 480},  # чуть нарушено
        {"N": 50, "M": 600},  # устойчиво
    ]

    run_test(params, grid_params_list)


def test_case_2():
    params = {
        'a': 1.0,
        'T': 2.0,
        'k': 0.1,
        'mu': lambda x: np.where((x > 0.4) & (x < 0.6), 1.0, 0.0),
        'mu1': lambda t: np.sin(2 * np.pi * t),
        'mu2': lambda t: np.cos(2 * np.pi * t),
        'title_mu1': 'u(0,t) = sin(2πt)',
        'title_mu2': 'u(1,t) = cos(2πt)',
        'title_case': 'Осциллирующие границы'
    }

    grid_params_list = [
        {"N": 50, "M": 100},  # сильно нарушено
        {"N": 50, "M": 997},  # чуть нарушено
        {"N": 50, "M": 1000},  # устойчиво
    ]

    run_test(params, grid_params_list)
