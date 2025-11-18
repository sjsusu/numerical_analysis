import numpy as np

def jacobi(A, b, x0=None, tol=1e-12, max_iterations=10000):
    x_prev = np.zeros_like(b) if x0 is None else x0.copy()
    diagonal_list = np.diag(A)
    D = np.diagflat(diagonal_list)
    R = A - D
    inverse_diag_list = 1.0 / diagonal_list
    D_inverse = np.diagflat(inverse_diag_list)

    for _ in range(max_iterations):
        rx_b = -np.dot(R, x_prev) + b
        x_next = np.dot(D_inverse, rx_b)
        if np.linalg.norm(x_next - x_prev, ord=np.inf) < tol:
            return x_next
        x_prev = x_next

    raise ValueError("Jacobi method did not converge within the maximum number of iterations")