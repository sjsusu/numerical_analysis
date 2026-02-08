import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def initial_condition(x):
    return np.sin(2 * np.pi * x)

def rk4_step(u, dt, dx, v):
    def F(u):
        # Compute spatial derivatives using central differences
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
        return -u * u_x + v * u_xx

    k1 = F(u)
    k2 = F(u + 0.5 * dt * k1)
    k3 = F(u + 0.5 * dt * k2)
    k4 = F(u + dt * k3)

    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def burgers_rk4(dx, t_final, v=0.2, start_x=0, end_x=1, return_history=False):
    # Take ceil for num_nodes
    num_nodes = int((end_x - start_x) / dx) + 1
    x = np.linspace(start_x, end_x, num_nodes)
    
    # The CFL condition for diffusion is more restrictive
    # as for initial condition u = sin(2pi x), max|u| = 1
    # Pick C_2 = 0.5 to be safe
    dt = 0.5 * dx**2 / v
    num_time_steps = int(np.ceil(t_final / dt))
    # Adjust dt to fit exactly into t_final
    dt = t_final / num_time_steps  

    u = initial_condition(x)

    # Store history of u for Question 3.4
    if return_history:
        t = 0
        u_history = [deepcopy(u)]
        times = [t]

    for _ in range(num_time_steps):
        u = rk4_step(u, dt, dx, v)
        if return_history:
            t += dt
            u_history.append(deepcopy(u))
            times.append(t)
            
    if return_history:
        return x, u, np.array(times), np.array(u_history)
    
    return x, u

def discrete_l2(u_num, u_ref, dx):
    return np.sqrt(np.sum((u_num - u_ref)**2) * dx)

def spacial_convergence(dx_list, t_final, v=0.2):
    
    # dx_exact will serve as our "exact" solution
    # Use a finer dx than minimum dx in dx_list
    dx_exact = min(dx_list) / 4
    x_exact, u_exact = burgers_rk4(dx_exact, t_final, v)
    
    errors = []
    for dx in dx_list:
        x_approx, u_approx = burgers_rk4(dx, t_final, v)
        # Interpolate exact solution to the current mesh of x_approx
        u_exact_on_x_approx = np.interp(x_approx, x_exact, u_exact)
        error = discrete_l2(u_approx, u_exact_on_x_approx, dx)
        errors.append(error)
    return dx_list, errors

def norm_vs_time(dx, t_final, x_exact, u_exact_history, v=0.2):
    x_approx, _, times, u_history = burgers_rk4(dx, t_final, v, return_history=True)
    
    norms = []
    for u_approx, u_exact in zip(u_history, u_exact_history):
        u_exact_on_x_approx = np.interp(x_approx, x_exact, u_exact)
        norm = discrete_l2(u_approx, u_exact_on_x_approx, dx)
        norms.append(norm)
    return times, norms

if __name__ == "__main__":
    dx = 1e-2
    t_final = 1
    x, u = burgers_rk4(dx, t_final)
    
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")

    plt.plot(x, u, label=f'$u(x, {t_final})$')
    plt.title("1D Burgers' Equation with RK4")
    plt.xlabel("$x$")
    plt.legend()
    plt.savefig('./outputs_3/burgers_rk4.png', dpi=300)
    plt.cla()
    
    dx_list = np.logspace(-2, -9, 18, base=2)
    dx, error = spacial_convergence(dx_list, t_final)
    plt.loglog(dx, error, label='$||u - u_h||_{L^2}$')
    plt.xlabel('$\\log(\\Delta x)$')
    plt.title('L2 Error vs Spatial Step Size')
    plt.legend()
    plt.savefig('./outputs_3/burgers_rk4_error.png', dpi=300)
    plt.show()
    
    dx_list = np.logspace(-2, -6, 5, base=2)

    dx_exact = min(dx_list) / 4
    x_exact, _, _, u_exact_history = burgers_rk4(dx_exact, t_final, return_history=True)

    for index, dx in enumerate(dx_list):
        time, norms = norm_vs_time(dx, t_final, x_exact, u_exact_history)
        plt.plot(time, norms, label=f'$h = 2^{{{-index-2}}}$')
    plt.xlabel('$t$')
    plt.ylabel(r'$||u - u_h||_{L^2}$')
    plt.title('L2 Error vs Time')
    plt.legend()
    plt.savefig('./outputs_3/burgers_rk4_error_vs_time.png', dpi=300)
    plt.show()