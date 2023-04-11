import numpy as np

from src.algorithms import gd, sd, cgd_direct, cgd_iter, cgd
from src.plot_utils import plot_quadratic_function


A = np.array([
        [4, 1],
        [1, 2]
    ])

b = np.array([
    [0],
    [2]
])

x0 = np.array([
    [2],
    [-2]
])

c = 10

def f(x):
    return (1/2)*(x.T @ A @ x) - b.T @ x + c

if __name__ == "__main__":
    plot_quadratic_function(f, title='cost_function')

    # Gradient Descent
    xs = gd(x0, A, b, 1e-1, 1e-6)
    plot_quadratic_function(f, xs, title='sgd')

    # Steepest Descent
    xs = sd(x0, A, b, 1e-6)
    plot_quadratic_function(f, xs, title='steepest_sgd')

    # Conjugant Gradient Descent via Direct Method
    xs = cgd_direct(A, b)
    plot_quadratic_function(f, xs, title='cgd_direct')

    # Conjugant Gradient Descent via Inefficient Iterative Method 
    xs = cgd_iter(x0, A, b)
    plot_quadratic_function(f, xs, title='cdg_iter_ineff')

    # Conjugant Gradient Descent via Iterative Method 
    xs = cgd(x0, A, b)
    plot_quadratic_function(f, xs, title='cgd')
