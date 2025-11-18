from assembly import global_assembly, global_indexing, classify_boundary_nodes, u_0, implement_dirichlet_bc
import numpy as np
import math
import matplotlib.pyplot as plt

def u_exact(x, y, t, boundary_indicator,nu=0.05):
    u = np.exp(-8 * (np.pi**2) * nu * t) * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    boundary_indicator = boundary_indicator.reshape(u.shape)
    u[boundary_indicator == 1] = 0.0  # Apply Dirichlet BCs
    return u

def implicit_heat_solver(x_nodes, dt, t_final, nu=0.05):
    global_coordinates, M, K = global_assembly(x_nodes)
    width = len(x_nodes)
    global_indexing_array = global_indexing(width)
    boundary_indicator = classify_boundary_nodes(global_indexing_array)

    M, K = implement_dirichlet_bc(M, K, boundary_indicator)

    A = M + nu * dt * K

    num_time_steps = math.ceil(t_final / dt)
    dt = t_final / num_time_steps
    prev_U = u_0(global_coordinates, boundary_indicator)
    U = np.array([prev_U])

    for n in range(num_time_steps):
        b = np.matmul(M, U[n])
        next_U = np.linalg.solve(A, b)
        U = np.append(U, [next_U], axis=0)

    return boundary_indicator, U.T, global_coordinates

def convergence_study_timesteps(x_nodes, t_final, dt_values, y_nodes=None):
    if y_nodes is None:
        y_nodes = x_nodes
    errors_implicit = []
    global_coordinates, _, _ = global_assembly(x_nodes, y_nodes)
    width = len(x_nodes)
    global_indexing_array = global_indexing(width)
    boundary = classify_boundary_nodes(global_indexing_array)
    x_coords = global_coordinates[:, 0].reshape(width**2, 1)
    y_coords = global_coordinates[:, 1].reshape(width**2, 1)
    u_exact_values = u_exact(x_coords, y_coords, t_final, boundary)

    for dt in dt_values:
        _, U, _ = implicit_heat_solver(x_nodes, dt, t_final)
        u_numerical_values = U[:, :, -1]
        error = np.linalg.norm(u_numerical_values - u_exact_values, ord=2)
        errors_implicit.append(error)

    return errors_implicit

if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")

    x_nodes = np.array([-2,-1.6, -1.2, -0.8, 2])
    # x_nodes = np.linspace(-2, 2, 8)
    width = len(x_nodes)
    dt = 0.01
    t_final = 1
    boundary, U, global_coordinates = implicit_heat_solver(x_nodes, dt, t_final)
    u_approx = U[:,:,-1].reshape((width, width))
    
    x_mesh = global_coordinates[:,0].reshape((width, width))
    y_mesh = global_coordinates[:,1].reshape((width, width))
    u_exact_values = u_exact(x_nodes, y_mesh, 1, boundary)

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x_mesh, y_mesh, u_approx, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('Numerical Solution')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x_mesh, y_mesh, u_exact_values, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title('Exact Solution')

    fig.savefig("./outputs_4/implicit_solution.png", dpi=300)
    # fig.savefig("./outputs_4/implicit_solution_uniform.png", dpi=300)
    # plt.show()
    
    dt_values = np.logspace(-1, -10, 10, base=2)
    errors_implicit = convergence_study_timesteps(x_nodes, t_final, dt_values)
    fig, ax = plt.subplots(figsize=(9,6))
    ax.loglog(dt_values, errors_implicit, marker='x', label='Implicit')
    ax.set_xlabel(r'$\log(\Delta t)$')
    ax.set_ylabel(r'$\log(||u - u_{exact}||_2)$')
    ax.set_title('Convergence Study for Implicit Solver')
    ax.legend()
    fig.savefig("./outputs_4/implicit_convergence_study.png", dpi=300, bbox_inches='tight')
