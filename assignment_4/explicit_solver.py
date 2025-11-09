from assembly import global_assembly
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt

def u_0(mesh):
    x = mesh[:, 0]
    y = mesh[:, 1]
    return np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

def u_exact(x, y, t, nu=0.05):
    return np.exp(-8 * (np.pi**2) * nu * t) * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

def explicit_heat_solver(width, dt, t_final, height=None, nu=0.05):
    if dt > 5/3:
        raise ValueError("Time step dt is too large for stability. Pick dt <= 5/3.")
    if height is None:
        height = width

    global_coordinates, M, K = global_assembly(width, height)
    M_inverse = np.linalg.inv(M)
    Identity = np.eye(M.shape[0])
    A = np.dot(M_inverse, K)
    B = Identity - nu * dt * A
    
    U = []
    t=0
    current_U = u_0(global_coordinates)
    U.append((t, deepcopy(current_U)))
    
    time_steps = math.ceil(t_final / dt)
    for _ in range(time_steps-1):
        t += dt
        current_U = np.dot(B, current_U)
        U.append((t, deepcopy(current_U)))
    
    dt = t_final - (time_steps-1)*dt
    B= Identity - nu * dt * A
    current_U = np.dot(B, current_U)
    U.append((t_final, deepcopy(current_U)))

    return U

if __name__ == "__main__":
    width = 5
    dt = 0.05
    t_final = 1.0
    U = explicit_heat_solver(width, dt, t_final)
    
    u_approx = U[-1][1].reshape((width-2, width-2))
    u_approx = np.pad(u_approx, (1, 1), 'constant', constant_values=(0,))
    x = np.linspace(-2, 2, width)
    x_interior = x[1:-1]
    y = np.linspace(-2, 2, width)
    y_interior = y[1:-1]
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_interior_mesh, y_interior_mesh = np.meshgrid(x_interior, y_interior)
    u_exact_values = u_exact(x_interior_mesh, y_interior_mesh, t_final)
    u_exact_values = np.pad(u_exact_values, (1, 1), 'constant', constant_values=(0,))

    fig = plt.figure()
    ax = fig.add_subplot( projection='3d')
    ax.plot_surface(x_mesh, y_mesh, u_approx, cmap='viridis', alpha=0.8, label='Approximate Solution')
    # ax.plot_surface(x_mesh, y_mesh, u_exact_values, cmap='plasma', alpha=0.3, label='Exact Solution')
    plt.show()