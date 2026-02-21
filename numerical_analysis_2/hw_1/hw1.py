import numpy as np
import matplotlib.pyplot as plt

# Constants
ALPHA = 1
BETA = 0.02
GAMMA = 0.6
DELTA = 0.01
TOLERANCE = 1e-10
Y_0 = np.array([50, 5])  # [x0, y0]


def f(Y):
    # Y = [x, y]
    x, y = Y
    dxdt = ALPHA * x - BETA * x * y
    dydt = -GAMMA * y + DELTA * x * y
    return np.array([dxdt, dydt])

def picard_step(y_prev_time, y_prev_iter, h):
    # Notation:
    # y_prev_time = [xn, yn] = [x_n, y_n]
    # y_prev_iter = [x_prev, y_prev] = [x_{n+1}^k, y_{n+1}^k]
    # y_new_iter = [x_new, y_new] = [x_{n+1}^{k+1}, y_{n+1}^{k+1}]

    def find_constants(y_prev_time, y_prev_iter):
        xn, yn = y_prev_time
        x_prev, y_prev = y_prev_iter

        constants = np.array([
            # Constant for x_n
            (1 + h * ALPHA / 2 - h * BETA * yn / 2),
            # Constant for x_{n+1}^k
            (h / 2) * (ALPHA - BETA * y_prev),
            # Constant for y_n
            (1 - h * GAMMA / 2 + h * DELTA * xn / 2),
            # Constant for y_{n+1}^k
            (h / 2) * (-GAMMA + DELTA * x_prev),
        ])

        return constants

    # Find constants for the linear combination of the previous time step and the previous iteration as noted in question 2
    c1, c2, c3, c4 = find_constants(y_prev_time, y_prev_iter)

    xn, yn = y_prev_time
    x_prev, y_prev = y_prev_iter

    # Compute the new values for x and y
    x_new = c1 * xn + c2 * x_prev
    y_new = c3 * yn + c4 * y_prev
    y_new_iter = np.array([x_new, y_new])

    error = max(abs(x_new - x_prev), abs(y_new - y_prev))

    return y_new_iter, error

def crank_nicolson_step(y, h):
    # Initial guess for the iteration
    y_prev_iter = y
    error = float("inf")
    iteration_count = 0

    # Fixed point iteration
    while error > TOLERANCE and iteration_count < 50:
        y_new_iter, error = picard_step(y, y_prev_iter, h)
        y_prev_iter = y_new_iter
        iteration_count += 1

    return y_new_iter

def crank_nicolson_solver(h, T=100):
    num_steps = int(T / h)
    y_values = np.zeros((num_steps + 1, 2))
    y_values[0] = Y_0

    # Perform time-stepping
    for n in range(num_steps):
        y_values[n + 1] = crank_nicolson_step(y_values[n], h)

    return y_values

if __name__ == "__main__":
    # Latex and Styling
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")
    # Get list of default colors for style
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    
    # --------------------------------------------------------
    # Question 3 and 4(a)
    # --------------------------------------------------------
    # Run the solver for different time steps and plot results
    h_values = [0.025, 0.05, .1, 0.5, 0.8]
    file_name_labels = ['0_025', '0_05', '0_1', '0_5', '0_8']
    
    for h, label in zip(h_values, file_name_labels):
        y_values = crank_nicolson_solver(h)
        time_values = np.arange(0, 100 + h, h)
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, y_values[:, 0], label=r"$x(t)$")
        plt.plot(time_values, y_values[:, 1], label=r"$y(t)$", color=default_colors[2])
        plt.xlabel(r"$t$")
        plt.title(rf"Crank-Nicolson Method ($\Delta t={h}$)")
        plt.legend()
        plt.savefig(f"./outputs/cn_{label}.png", dpi=300)
        plt.cla()
    
    # --------------------------------------------------------
    # Question 4(b)
    # --------------------------------------------------------
    # Phase plots for h = 0.025 and h = 0.5
    h_values = [0.025, 0.5]
    file_name_labels = ['0_025', '0_5']
    for h, label in zip(h_values, file_name_labels):
        y_values = crank_nicolson_solver(h)
        plt.figure(figsize=(8, 8))
        plt.plot(y_values[:, 0], y_values[:, 1], label=rf"$\Delta t={h}$")
        plt.xlabel(r"$x(t)$")
        plt.ylabel(r"$y(t)$")
        plt.title(rf"Phase Plot ($\Delta t={h}$)")
        plt.legend()
        plt.savefig(f"./outputs/cn_phase_{label}.png", dpi=300)
        plt.cla()
