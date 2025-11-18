from assembly import global_assembly
from assignment_4.implicit_euler import u_0, u_exact, explicit_heat_solver
import numpy as np
import math
import matplotlib.pyplot as plt

def jacobi(A, b, x0=None, tol=1e-12, max_iterations=10000):
    x_prev = np.zeros_like(b) if x0 is None else x0.copy()
    diagonal_list = np.diag(A)
    D = np.diagflat(diagonal_list)
    R = A - D
    D_inverse = np.linalg.inv(D)

    for _ in range(max_iterations):
        rx_b = -np.dot(R, x_prev) + b
        x_next = np.dot(D_inverse, rx_b)
        if np.linalg.norm(x_next - x_prev, ord=np.inf) < tol:
            return x_next
        x_prev = x_next

    raise ValueError("Jacobi method did not converge within the maximum number of iterations")

def crank_nicolson(x_nodes, t_final, dt, y_nodes=None, nu=0.05, use_jacobi=True):
    if y_nodes is None:
        y_nodes = x_nodes
    global_coordinates, M, K = global_assembly(x_nodes, y_nodes)
    num_nodes = M.shape[0]
    num_time_steps = math.ceil(t_final / dt)
    dt = t_final / num_time_steps

    U = np.zeros((num_nodes, num_time_steps))
    U[:, 0] = u_0(global_coordinates)

    A = M + (nu * dt / 2) * K
    B = M - (nu * dt / 2) * K

    for n in range(num_time_steps-1):
        b = np.dot(B, U[:, n])
        if use_jacobi:
            U[:, n+1] = jacobi(A, b, x0=U[:, n])
        else:
            U[:, n+1] = np.linalg.solve(A, b)

    times = np.array([n * dt for n in range(num_time_steps)])

    return times, U.T


def convergence_study_timesteps(x_nodes, t_final, dt_values, y_nodes=None):
    if y_nodes is None:
        y_nodes = x_nodes
    errors_crank = []
    errors_explicit = []
    global_coordinates, _, _ = global_assembly(x_nodes, y_nodes)
    u_exact_values = u_exact(global_coordinates[:, 0], global_coordinates[:, 1], t_final)
    for dt in dt_values:
        _, U = crank_nicolson(x_nodes, t_final, dt, y_nodes=y_nodes, use_jacobi=False)

        u_numerical_values = U[-1, :]
        error = np.linalg.norm(u_numerical_values - u_exact_values, ord=2)
        errors_crank.append(error)

    for dt in dt_values:
        _, U = explicit_heat_solver(x_nodes, dt, t_final)
        u_numerical_values = U[-1, :]
        error = np.linalg.norm(u_numerical_values - u_exact_values, ord=2)
        errors_explicit.append(error)

    return errors_crank, errors_explicit

if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")
    
    width = np.array([-2, -1.6, -1.2, -0.8, 2])
    t_final = 1
    dt_values = np.logspace(-1, -2, 4)

    errors_crank, errors_explicit = convergence_study_timesteps(width, t_final, dt_values)
    
    fig, ax = plt.subplots(figsize=(9,6))
    ax.loglog(dt_values, errors_crank, marker='o', label='Crank-Nicolson')
    ax.loglog(dt_values, errors_explicit, marker='x', label='Explicit')
    ax.set_xlabel(r'$\\log(\Delta t)$')
    ax.set_ylabel(r'$\\log(|u - u_{exact}|_2)$')
    ax.set_title(rf'Convergence Study (width$={width})$')
    ax.legend()
    fig.savefig(f'./outputs_4/convergence_width_{width}.png', dpi=300)
    plt.close(fig)
        
    width=6
    dt = 1e-3
    times, U = crank_nicolson(width, t_final, dt, use_jacobi=False)
    u_approx = U[-1].reshape((width-2, width-2))
    u_approx = np.pad(u_approx, (1, 1), 'constant', constant_values=(0,))
    
    _, U_explicit = explicit_heat_solver(width, dt, t_final)
    u_approx_explicit = U_explicit[-1].reshape((width-2, width-2))
    u_approx_explicit = np.pad(u_approx_explicit, (1, 1), 'constant', constant_values=(0,))
    x = np.linspace(-2, 2, width)
    x_interior = x[1:-1]
    y = np.linspace(-2, 2, width)
    y_interior = y[1:-1]
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_interior_mesh, y_interior_mesh = np.meshgrid(x_interior, y_interior)
    u_exact_values = u_exact(x_interior_mesh, y_interior_mesh, t_final)
    u_exact_values = np.pad(u_exact_values, (1, 1), 'constant', constant_values=(0,))
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(x_mesh, y_mesh, u_approx, cmap='viridis', alpha=0.8)
    ax1.set_title('Crank-Nicolson Approximate Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u_h(x,y,t_{final})')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(x_mesh, y_mesh, u_exact_values, cmap='plasma', alpha=0.8)
    ax2.set_title('Exact Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y,t_{final})')

    ax3 = fig.add_subplot(2, 2, 1, projection='3d')
    u_approx_explicit = U_explicit[-1].reshape((width-2, width-2))
    u_approx_explicit = np.pad(u_approx_explicit, (1, 1), 'constant', constant_values=(0,))
    ax3.plot_surface(x_mesh, y_mesh, u_approx_explicit, cmap='viridis', alpha=0.8)
    ax3.set_title('Explicit Approximate Solution')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u_h(x,y,t_{final})')    
    plt.show()