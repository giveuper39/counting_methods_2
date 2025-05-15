import numpy as np
from scipy.optimize import minimize
from time import time


def numerical_gradient(f, x, eps=1e-6):
    """Вычисляет численный градиент с центральными разностями"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def project(x, bounds):
    """Проекция точки на ограничения"""
    return np.clip(x, [b[0] if b[0] is not None else -np.inf for b in bounds],
                   [b[1] if b[1] is not None else np.inf for b in bounds])


def projected_gradient_descent(f, x0, bounds, constraints_ineq=None,
                               constraints_eq=None, lr=0.001, max_iter=10000, eps=1e-6):
    """Метод проекции градиента"""
    x = project(np.array(x0, dtype=float), bounds)
    fevals = 0
    best_x = x.copy()
    best_f = np.inf

    def F(x):
        nonlocal fevals
        val = f(x)
        fevals += 1

        # Штраф за неравенства
        for g in constraints_ineq or []:
            violation = max(0, -g(x))
            val += 1000 * violation ** 2

        # Штраф за равенства
        for h in constraints_eq or []:
            val += 1000 * h(x) ** 2

        return val

    start_time = time()

    for iteration in range(max_iter):
        grad = numerical_gradient(F, x)
        fevals += len(x) * 2

        # Адаптивный learning rate
        current_lr = lr
        for _ in range(10):
            x_new = project(x - current_lr * grad, bounds)
            if F(x_new) < F(x):
                break
            current_lr *= 0.5

        if np.linalg.norm(x_new - x) < eps:
            break

        x = x_new
        current_f = f(x)
        if current_f < best_f:
            best_f = current_f
            best_x = x.copy()

    end_time = time()
    return best_x, best_f, fevals, end_time - start_time


def penalty_method(f, x0, constraints_eq, constraints_ineq, bounds=None,
                   mu0=10.0, rho=2.0, max_iter=50, tol=1e-6):
    """Метод штрафов"""
    bounds = bounds or [(None, None)] * len(x0)
    x = np.array(x0, dtype=float)
    mu = mu0
    fevals = 0
    start_time = time()

    for k in range(max_iter):
        def P(x):
            val = f(x)
            for g in constraints_eq:
                val += mu * g(x) ** 2
            for h in constraints_ineq:
                val += mu * max(0, -h(x)) ** 2
            return val

        res = minimize(P, x, bounds=bounds, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-8})
        x = res.x
        fevals += res.nfev

        # Проверка ограничений
        eq_violation = sum(abs(g(x)) for g in constraints_eq)
        ineq_violation = sum(max(0, -h(x)) for h in constraints_ineq)

        if eq_violation + ineq_violation < tol:
            break

        mu *= rho

    end_time = time()
    return x, f(x), fevals, end_time - start_time


def augmented_lagrangian(f, x0, constraints_eq, constraints_ineq, bounds=None,
                         max_iter=50, tol=1e-6):
    """Исправленный метод модифицированных функций Лагранжа"""
    bounds = bounds or [(None, None)] * len(x0)
    x = np.array(x0, dtype=float)
    fevals = 0

    # Инициализация параметров
    lambda_eq = np.zeros(len(constraints_eq))
    lambda_ineq = np.zeros(len(constraints_ineq))
    mu = 10.0  # Начальный параметр штрафа
    mu_max = 1e6  # Максимальное значение параметра штрафа
    rho = 1.5  # Коэффициент увеличения штрафа

    best_x = x.copy()
    best_f = np.inf
    start_time = time()

    for k in range(max_iter):
        # Модифицированная функция Лагранжа
        def L(x):
            nonlocal fevals
            val = f(x)
            fevals += 1

            # Ограничения-равенства
            for i, g in enumerate(constraints_eq):
                c = g(x)
                val += lambda_eq[i] * c + (mu/2) * c**2

            # Ограничения-неравенства
            for j, h in enumerate(constraints_ineq):
                c = h(x)
                val += lambda_ineq[j] * c + (mu/2) * min(0, c)**2

            return val

        # Минимизация
        res = minimize(L, x, bounds=bounds, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-8})
        x = res.x
        fevals += res.nfev

        # Вычисление нарушений ограничений
        eq_violations = [g(x) for g in constraints_eq]
        ineq_violations = [h(x) for h in constraints_ineq]

        # Обновление множителей
        for i, c in enumerate(eq_violations):
            lambda_eq[i] += mu * c

        for j, c in enumerate(ineq_violations):
            lambda_ineq[j] = max(0, lambda_ineq[j] + mu * min(0, c))

        # Проверка сходимости
        max_eq_viol = max(abs(c) for c in eq_violations) if constraints_eq else 0
        max_ineq_viol = max(max(0, -c) for c in ineq_violations) if constraints_ineq else 0

        if max(max_eq_viol, max_ineq_viol) < tol:
            break

        # Увеличение параметра штрафа
        if max(max_eq_viol, max_ineq_viol) > 0.25 * tol:
            mu = min(mu * rho, mu_max)

        # Сохранение лучшего решения
        current_f = f(x)
        if current_f < best_f and max(max_eq_viol, max_ineq_viol) < 10*tol:
            best_f = current_f
            best_x = x.copy()

    end_time = time()
    return best_x, best_f, fevals, end_time - start_time


def run_test(name, f, x0, bounds, true_min, constraints_eq=None, constraints_ineq=None,
             methods=(projected_gradient_descent, penalty_method, augmented_lagrangian)):
    print(f"\n{'=' * 40}")
    print(f"ТЕСТ: {name}")
    print(f"{'=' * 40}")

    results = []

    for method in methods:
        try:
            args = {
                'f': f,
                'x0': x0,
                'bounds': bounds
            }

            if method == projected_gradient_descent:
                args['constraints_ineq'] = constraints_ineq or []
                args['constraints_eq'] = constraints_eq or []
            elif method in (penalty_method, augmented_lagrangian):
                args['constraints_eq'] = constraints_eq or []
                args['constraints_ineq'] = constraints_ineq or []

            x_opt, f_opt, fevals, t = method(**args)

            error = abs(f_opt - true_min)
            results.append({
                'method': method.__name__,
                'x': x_opt,
                'f': f_opt,
                'error': error,
                'fevals': fevals,
                'time': t
            })

            print(f"{method.__name__}:")
            print(f"  Решение x = {np.round(x_opt, 6)}")
            print(f"  Значение f(x) = {f_opt:.6f}")
            print(f"  Число вычислений f = {fevals}")
            print(f"  Время = {t:.4f} сек")
            print(f"  Точное решение: f* = {true_min:.6f}")
            print(f"  Погрешность = {error:.6f}")
        except Exception as e:
            print(f"{method.__name__} FAILED: {str(e)}")

    return results