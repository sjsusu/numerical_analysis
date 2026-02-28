import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def F(u, sigma, rho, beta):
    x, y, z = u
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def RK4_step(u, h, sigma, rho, beta):
    k1 = F(u, sigma, rho, beta)
    k2 = F(u + (h / 2) * k1, sigma, rho, beta)
    k3 = F(u + (h / 2) * k2, sigma, rho, beta)
    k4 = F(u + h * k3, sigma, rho, beta)

    y_new = u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_new


def RK4_solver(u_0, h, sigma, rho, beta, T=50):
    num_steps = int(T / h)
    u_values = np.zeros((num_steps + 1, len(u_0)))
    u_values[0] = u_0

    for n in range(num_steps):
        u_values[n + 1] = RK4_step(u_values[n], h, sigma, rho, beta)

    return u_values


if __name__ == "__main__":
    # Latex and Styling
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")
    # Get list of default colors for style
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]

    # ---------------------------------------------
    # Question 2
    # ---------------------------------------------
    # Use default parameters for Lorenz system (Case C)
    sigma, rho, beta = 10, 28, 8 / 3
    u_0 = np.array([1, 1, 1])
    dt = 0.01
    time_steps = [dt, dt / 2, dt / 4]

    # Run RK4 solver for each time step and plot results
    for h in time_steps:
        u_values = RK4_solver(u_0, h, sigma, rho, beta)
        time_values = np.arange(0, 50 + h, h)
        plt.plot(time_values, u_values[:, 0], label=rf"$\Delta t = {h}$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.title("RK4 Results")
    plt.legend()
    plt.savefig("./outputs/rk4_q2.png", dpi=300)
    plt.cla()

    # Compare with Scipy's ODE solver (similar to ode45 for matlab)
    scipy_solution = solve_ivp(
        lambda t, u: F(u, sigma, rho, beta),
        [0, 50],
        u_0,
        t_eval=np.arange(0, 50 + dt, dt),
    )

    # Plot Scipy solution
    plt.plot(
        scipy_solution.t, scipy_solution.y[0], color=default_colors[3]
    )
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.title("Scipy Solver Results")
    plt.savefig("./outputs/rk4_scipy.png", dpi=300)
    plt.cla()

    labels = ["1", "2", "3"]
    color = 0
    for h, label in zip(time_steps, labels):
        u_values = RK4_solver(u_0, h, sigma, rho, beta)
        time_values = np.arange(0, 50 + h, h)
        plt.plot(
            scipy_solution.t,
            scipy_solution.y[0],
            label="Scipy",
            color=default_colors[3],
        )
        plt.plot(
            time_values,
            u_values[:, 0],
            label=rf"RK4 $\Delta t = {h}$",
            color=default_colors[color],
        )
        plt.title("RK4 vs Scipy Solver")
        plt.legend()
        plt.savefig(f"./outputs/rk4_comparison_{label}.png", dpi=300)
        plt.cla()
        color += 1

    # ---------------------------------------------
    # Question 3
    # ---------------------------------------------
    # each element of param_sets is [sigma, rho, beta]
    param_sets = np.array(
        [
            # Case A
            [10, 5, 8 / 3],
            # Case B
            [10, 15, 8 / 3],
            # Case C
            [10, 28, 8 / 3],
            # Case D
            [10, 40, 8 / 3],
        ]
    )
    
    set_labels = ["A", "B", "C", "D"]
    
    u_0 = np.array([1, 1, 1])
    h = 0.01 # baseline time step
    time_values = np.arange(0, 50 + h, h)
    dimensions = ["x", "y", "z"]
    
    # Generate plots for each case
    for params, label in zip(param_sets, set_labels):
        u_values = RK4_solver(u_0, h, *params)
        
        # Plot x(t), y(t), z(t) individually 
        for i, dim in enumerate(dimensions):
            fig, ax = plt.subplots()
            ax.plot(time_values, u_values[:, i], label=f"${dim}(t)$", color=default_colors[i])
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(rf"${dim}(t)$")
            ax.set_title(rf'RK4 for Case {label}: $(\sigma, \rho, \beta) = (10, {int(params[1])}, \frac{8}{3})$')
            ax.legend()
            fig.savefig(rf"./outputs/rk4_{label}_{dim}.png", dpi=300)
            plt.close(fig)
            
        # Plot Phase Portrait
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(u_values[:, 0], u_values[:, 1], u_values[:, 2], color=default_colors[3])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        plt.title(rf'Phase Portrait for Case {label}: $(\sigma, \rho, \beta) = (10, {int(params[1])}, \frac{8}{3})$')
        plt.savefig(rf"./outputs/rk4_{label}_phase_portrait.png", dpi=300)
        plt.close(fig)
    
    # ---------------------------------------------
    # Question 4
    # ---------------------------------------------
    
    initial_conditions = np.array([
        [1,1,1],
        [1+1e-8, 1, 1]
    ])
    dimensions = ["x", "y", "z"]
    ic_labels = ["original", "perturbed"]
    
    # Pick baseline dt
    h = 0.01
    # Use Case C parameters for this question
    sigma, rho, beta = 10, 28, 8/3
    
    # Run for each dimension
    for i in range(3):
        # Store dim(t) values for both initial conditions
        dim_values = []
        
        # Plot dim(t) for both initial conditions on same graph
        for u_0, label in zip(initial_conditions, ic_labels):
            u_values = RK4_solver(u_0, h, sigma, rho, beta)
            time_values = np.arange(0, 50 + h, h)
            plt.plot(time_values, u_values[:, i], label=f"{label}")
            dim_values.append(u_values[:, i])
            
        plt.xlabel(r"$t$")
        plt.ylabel(rf"${dimensions[i]}(t)$")
        plt.title(rf"Initial Condition Comparison for ${dimensions[i]}(t)$")
        plt.legend()
        plt.savefig(f"./outputs/q4/rk4_ic_{dimensions[i]}.png", dpi=300)
        plt.cla()
        
        # Plot |dim_1(t) - dim_2(t)| on semilog-y graph
        diff = np.abs(dim_values[1] - dim_values[0])
        plt.semilogy(time_values, diff, label=rf"$|{dimensions[i]}_1(t) - {dimensions[i]}_2(t)|$")
        plt.xlabel(r"$t$")
        plt.ylabel(rf"$|{dimensions[i]}_1(t) - {dimensions[i]}_2(t)|$")
        plt.title(rf"Absolute Difference for ${dimensions[i]}(t)$")
        plt.legend()
        plt.savefig(f"./outputs/q4/rk4_ic_diff_{dimensions[i]}.png", dpi=300)
        plt.cla()
        
    # ---------------------------------------------
    # Question 5
    # ---------------------------------------------
    
    time_steps = [0.02, 0.01, 0.005]
    file_labels = ["_02", "_01", "_005"]
    dimensions = ["x", "y", "z"]
    
    # Use Case D parameters
    sigma, rho, beta = 10, 40, 8/3
    u_0 = np.array([1, 1, 1])
    
    # Generate plots for each time step
    for h, label in zip(time_steps, file_labels):
        u_values = RK4_solver(u_0, h, sigma, rho, beta)
        time_values = np.arange(0, 50 + h, h)
        
        # Generate time plots for each dimension
        for i, dim in enumerate(dimensions):
            fig, ax = plt.subplots()
            ax.plot(time_values, u_values[:, i], label=f"${dim}(t)$", color=default_colors[i])
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(rf"${dim}(t)$")
            ax.set_title(rf'RK4 for Case D with $\Delta t = {h}$')
            ax.legend()
            fig.savefig(rf"./outputs/q5/rk4_{dim}{label}.png", dpi=300)
            plt.close(fig)
            
        # Generate phase portrait
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(u_values[:, 0], u_values[:, 1], u_values[:, 2], color=default_colors[3])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title(rf'Phase Portrait for Case D with $\Delta t = {h}$')
        fig.savefig(rf"./outputs/q5/rk4_phase_portrait{label}.png", dpi=300)
        plt.close(fig)