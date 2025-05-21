import numpy as np

from task7.main import main_loop

def simple_odu_test():
    def exact_solution(x):
        return -np.cos(x)/np.cos(1) - x**2 + 2
    
    def p(x): return 0
    def q(x): return 1
    def f(x): return -x**2
    
    main_loop([-1, 1], 'Lu = u\'\' + u = -x²', p, q, f, 3, exact_solution)
    main_loop([-1, 1], 'Lu = u\'\' + u = -x²', p, q, f, 4, exact_solution)
    main_loop([-1, 1], 'Lu = u\'\' + u = -x²', p, q, f, 5, exact_solution)

def trigonometric_ode_test():
    def exact_solution(x):
        C1 = 0
        C2 = np.cos(2)/(4*np.sin(2))
        return C1*np.cos(2*x) + C2*np.sin(2*x) - (x*np.cos(2*x))/4
    
    def p(x): return 0
    def q(x): return 4
    def f(x): return np.sin(2*x)
    
    main_loop([-1, 1], 'Lu = u\'\' + 4u = sin(2x)', p, q, f, 2, exact_solution)
    main_loop([-1, 1], 'Lu = u\'\' + 4u = sin(2x)', p, q, f, 3, exact_solution)
    main_loop([-1, 1], 'Lu = u\'\' + 4u = sin(2x)', p, q, f, 5, exact_solution)

def exponential_ode_test():
    def exact_solution(x):
        A = np.array([
            [np.cos(1), -np.sin(1)], 
            [np.cos(1), np.sin(1)]
        ])
        b = np.array([-0.5*np.exp(-1), -0.5*np.exp(1)])
        C1, C2 = np.linalg.solve(A, b)
        return C1*np.cos(x) + C2*np.sin(x) + 0.5*np.exp(x)
    
    def p(x): return 0
    def q(x): return 1
    def f(x): return np.exp(x)
    
    main_loop([-1, 1], 'Lu = u\'\' + u = e^x', p, q, f, 1, exact_solution)
    main_loop([-1, 1], 'Lu = u\'\' + u = e^x', p, q, f, 2, exact_solution)
    main_loop([-1, 1], 'Lu = u\'\' + u = e^x', p, q, f, 3, exact_solution)

# trigonometric_ode_test()
# simple_odu_test()
# exponential_ode_test()

# Когда точное решение не выражается в элементарных функциях

# def complex_odu_test():
#     def p2(x): return x
#     def q2(x): return np.exp(-x**2)
#     def f2(x): return np.sin(np.pi*x)
    
#     main_loop([-1, 1], 'Lu = u\'\' + x*u\' + e^(-x²)u = sin(πx)', p2, q2, f2)
#     main_loop([-1, 1], 'Lu = u\'\' + x*u\' + e^(-x²)u = sin(πx)', p2, q2, f2, 4)
#     main_loop([-1, 1], 'Lu = u\'\' + x*u\' + e^(-x²)u = sin(πx)', p2, q2, f2, 5)

# def volcov_book_test_nodes():
#     def p(x): return 0
#     def q(x): return 1 + x**2
#     def f(x): return -1
    
#     main_loop([-1, 1], 'Lu = u'' + (1 + x^2)u = -1', p, q, f)
#     main_loop([-1, 1], 'Lu = u'' + (1 + x^2)u = -1', p, q, f, 4)
#     main_loop([-1, 1], 'Lu = u'' + (1 + x^2)u = -1', p, q, f, 5)

# complex_odu_test()
# volcov_book_test_nodes()