from scipy.special import jacobi
from scipy.integrate import quad
import numpy as np
from scipy.linalg import solve
from scipy.special import roots_chebyt
import matplotlib.pyplot as plt

"""
Решает ОДУ: y'' + p(x)y' + q(x)y = f(x) с y(a)=y(b)=0 (краевое условие первого рода)
    
Параметры:
    p, q, f - функции коэффициентов
    a, b - границы интервала
    n - число базисных функций

Методы: проекционный - коллокации, вариационный -  Ритца
"""

def solve_collocation(graphic_boundaries, p, q, f, n=5):
    a = graphic_boundaries[0]
    b = graphic_boundaries[1]
    
    def phi(k, x):
        Tk = np.polynomial.chebyshev.Chebyshev.basis(k)
        return (b - x)*(x - a)/((b-a)**2) * Tk((2*x - (a+b))/(b-a))
    
    def dphi(k, x):
        Tk = np.polynomial.chebyshev.Chebyshev.basis(k)
        dTk = Tk.deriv()
        t = (2*x - (a+b))/(b-a)
        term1 = (a + b - 2*x)/(b-a)**2 * Tk(t)
        term2 = (b-x)*(x-a)/(b-a)**2 * dTk(t) * 2/(b-a)
        return term1 + term2
    
    def d2phi(k, x):
        Tk = np.polynomial.chebyshev.Chebyshev.basis(k)
        dTk = Tk.deriv()
        d2Tk = Tk.deriv(2)
        t = (2*x - (a+b))/(b-a)
        term1 = -2/(b-a)**2 * Tk(t)
        term2 = 2*(a + b - 2*x)/(b-a)**3 * dTk(t) * 2
        term3 = (b-x)*(x-a)/(b-a)**2 * d2Tk(t) * (2/(b-a))**2
        return term1 + term2 + term3
    
    nodes = roots_chebyt(n)[0] * (b-a)/2 + (a+b)/2
    
    A = np.zeros((n, n))
    b_vec = np.zeros(n)
    
    for i in range(n):
        x = nodes[i]
        for k in range(n):
            A[i, k] = d2phi(k,x) + p(x)*dphi(k,x) + q(x)*phi(k,x)
        b_vec[i] = f(x)
    
    c = solve(A, b_vec)
    x_plot = np.linspace(a, b, 200)
    y_plot = np.zeros_like(x_plot)
    for k in range(n):
        y_plot += c[k] * phi(k, x_plot)
        
    return x_plot, y_plot

def solve_ritz(graphic_boundaries, p, q, f, n=15):
    alpha, beta = 0, 0
    
    a = graphic_boundaries[0]
    b = graphic_boundaries[1]
    def phi(k, x):
        Pk = jacobi(k, alpha, beta)
        return (x - a)*(b - x) * Pk((2*x - (a+b))/(b-a))
    
    def dphi(k, x):
        Pk = jacobi(k, alpha, beta)
        dPk = Pk.deriv()
        t = (2*x - (a+b))/(b-a)
        term1 = (a + b - 2*x) * Pk(t)
        term2 = (x-a)*(b-x) * dPk(t) * 2/(b-a)
        return term1 + term2
    
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    for i in range(n):
        B[i] = quad(lambda x: f(x) * phi(i, x), a, b)[0]
        
        for j in range(n):
            integrand = lambda x: -dphi(i,x)*dphi(j,x) + p(x)*dphi(i,x)*phi(j,x) + q(x)*phi(i,x)*phi(j,x)
            A[i][j] = quad(integrand, a, b)[0]
    
    c = solve(A, B)
    
    x_plot = np.linspace(a, b, 200)
    y_plot = np.zeros_like(x_plot)
    
    for k in range(n):
        y_plot += c[k] * phi(k, x_plot)
    
    return x_plot, y_plot

def exact_solution_plot(graphic_boundaries, n, exact_solution):
    x_plot = np.linspace(graphic_boundaries[0], graphic_boundaries[1], 200)
    y_plot = np.zeros_like(x_plot)
    y_plot += exact_solution(x_plot)
    return x_plot, y_plot

def show_solutions(equation_string, n, solutions):
    x_collocation, y_collocation = solutions[0]
    x_ritz, y_ritz = solutions[1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_collocation, y_collocation, 'b--', linewidth=2, label='Коллокация (Чебышёв)')
    plt.plot(x_ritz, y_ritz, 'r--', linewidth=2, label='Ритц (Якоби)')
    
    if (len(solutions) > 2):
        x_exact, y_exact = solutions[2]
        plt.plot(x_exact, y_exact, ':', color='#00BFFF', linewidth=2, label='Точное решение')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title(f'Сравнение решений разными методами уравнения: {equation_string}', fontsize=14)
    plt.text(0.02, 0.98, f'Число узлов: {n}\n', 
         ha='left', va='top', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, framealpha=1)
    plt.tight_layout()
    plt.show()
    
    
def main_loop(graphic_boundaries, equation_string, p, q, f, n = 3, exact_solution = lambda x: '1'):
    
    solutions = [solve_collocation(graphic_boundaries, p, q, f, n), solve_ritz(graphic_boundaries, p, q, f, n)]
    
    if (exact_solution(0) != '1'):
        solutions.append(exact_solution_plot(graphic_boundaries, n, exact_solution))
    
    show_solutions(equation_string, n, solutions)
    
