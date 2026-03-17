from rk4 import RK4_solver
from secant import secant_method
import numpy as np
import matplotlib.pyplot as plt

Y_0 = 0
Y_RIGHT_BC = 2

def exact_solution(x):
    return 2 * np.sin(x)

def F(u):
    y, z = u
    return np.array([z, -y])

# residual function is F(s)
def residual(s, h=0.01):
    u_0 = np.array([Y_0, s])
    y_values = RK4_solver(F, u_0, h=h)

    # Return F(s) = y(T) - Y_RIGHT_BC
    return y_values[-1, 0] - Y_RIGHT_BC

def generate_tables(s_values, F_values, path):
    table_line = r"\hline" + "\n"
    with open(path, "w") as file:
        # Add final s value and F(s)
        # Determine final index for s and F(s) values
        n_final = len(s_values) - 1
        total_iterations = n_final - 1 
        file.write(f"Total iterations: {total_iterations}\n")
        file.write(f"$s_{{{n_final}}}$: {s_values[-1]:.6f}\n")
        file.write(f"$F(s_{{{n_final}}})$: {F_values[-1]:.6e}\n\n")      

        # First table with inital s values and their corresponding F(s) values
        # n | s_n | F(s_n)
        file.write( r"\begin{center}" + "\n")
        file.write(r"\begin{tabular}{|c|c|c|}" + "\n")
        file.write(table_line)
        file.write(r"$n$ & $s_n$ & $F(s_n)$ \\" + "\n")
        file.write(table_line)
        
        for i in range(2):
            s, F_s = s_values[i], F_values[i]
            file.write(rf"${i}$ & ${s:.6f}$ & ${F_s:.6e}$ \\" + "\n")
            file.write(table_line)
            
        file.write(r"\end{tabular}" + "\n")
        file.write(r"\end{center}" + "\n")
        
        file.write("\n\n") # Add some space between the tables
        
        # Second table with secant method iterations
        # n | s_{n-1} | s_n | s_{n+1} | F(s_{n+1})
        file.write(r"\begin{center}" + "\n")
        file.write(r"\begin{tabular}{|c|c|c|c|c|}" + "\n")
        file.write(table_line)
        file.write(r"$n$ & $s_{n-1}$ & $s_n$ & $s_{n+1}$ & $F(s_{n+1})$ \\" + "\n")
        file.write(table_line)
        
        for n in range(1, len(s_values) - 1): 
            s_past, s_current, s_next = s_values[n-1], s_values[n], s_values[n + 1]
            F_next = F_values[n + 1]
            file.write(
                rf"${n}$ & ${s_past:.6f}$ & ${s_current:.6f}$ & ${s_next:.6f}$ & ${F_next:.6e}$ \\" + "\n"
            )
            file.write(table_line)
        file.write(r"\end{tabular}" + "\n")
        file.write(r"\end{center}" + "\n")
    return

if __name__ == "__main__":
    
    # ------------------------------------------
    # Question 2
    # -------------------------------------------
    # Used for finding initial s_0 and s_1
    s_values = [4 / np.pi, 4/ np.pi + 1]  
    for s in s_values:
        print(f"s = {s}, F(s) = {residual(s)}")
    
    # Run shooting method
    s0, s1 =  4 / np.pi, 4/ np.pi + 1
    s_values, F_values = secant_method(residual, s0, s1)
    generate_tables(s_values, F_values, "./outputs/tables.txt")
    
    # ------------------------------------------
    # Question 3
    # -------------------------------------------
    # L2 Error Check
    s_final = s_values[-1]
    h = 0.01
    u_0 = np.array([Y_0, s_final])
    y_values = RK4_solver(F, u_0, h=h)
    x_values = np.linspace(0, np.pi / 2, len(y_values))
    y_exact = exact_solution(x_values)
    l2_error = np.sqrt(np.sum((y_values[:, 0] - y_exact) ** 2))
    print(f"L2 error between numerical and exact solution: {l2_error:.6e}")
    
    # ------------------------------------------
    # Question 4
    # -------------------------------------------
    # Step size refinement and order of accuracy
    
    # ------------------------------------------
    # (a) Generate table for step size refinement
    # ------------------------------------------
    s0, s1 =  4 / np.pi, 4/ np.pi + 1
    pi = np.pi
    h_values = [pi/10, pi/20, pi/40, pi/80]
    s_values = []
    F_values = []
    l2_values = []
    
    # Compute final s, F(s), and L2 error for each h
    for h in h_values:
        s_values, F_values = secant_method(residual, s0, s1, h=h)
        s_final = s_values[-1]
        u_0 = np.array([Y_0, s_final])
        y_values = RK4_solver(F, u_0, h=h)
        x_values = np.linspace(0, pi/2, len(y_values))
        y_exact = exact_solution(x_values)
        l2_error = np.sqrt(np.sum((y_values[:, 0] - y_exact) ** 2))
        
        s_values.append(s_final)
        F_values.append(F_values[-1])
        l2_values.append(l2_error)
    
    # Write table to file
    # h | s_final | F(s_final) | L2 Error
    with open("./outputs/step_size_study.txt", "w") as file:
        table_line = r"\hline" + "\n"
        
        file.write(r"\begin{center}" + "\n")
        file.write(r"\begin{tabular}{|c|c|c|c|}" + "\n")
        file.write(table_line)
        file.write(r"$h$ & $s$ & $|F(s)|$ & $e_{L^2}(h)$ \\" + "\n")
        file.write(table_line)
        
        for h, s, F_final, l2 in zip(h_values, s_values, F_values, l2_values):
            F_final = abs(F_final)
            file.write(
                rf"${h:.6f}$ & ${s:.6f}$ & ${F_final:.6e}$ & ${l2:.6e}$ \\" + "\n"
            )
            file.write(table_line)
        
        file.write(r"\end{tabular}" + "\n")
        file.write(r"\end{center}" + "\n")
    
    # Latex and Styling
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")
    # Get list of default colors for style
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    
    # ------------------------------------------
    # (b) Plot y* and y for h = pi/160
    # ------------------------------------------
    h = pi/160
    s_values, F_values = secant_method(residual, s0, s1, h=h)
    s_final = s_values[-1]
    u_0 = np.array([Y_0, s_final])
    y_values = RK4_solver(F, u_0, h=h)
    x_values = np.linspace(0, pi/2, len(y_values))
    y_exact = exact_solution(x_values)
    
    plt.plot(x_values, y_values[:, 0], label=r"$y^*(x)$", color=default_colors[0])
    plt.plot(x_values, y_exact, label=r"$y(x)$", color=default_colors[2])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y(x)$")
    plt.title(r"Numerical Solution $y^*(x)$ vs Exact Solution $y(x)$ ($h = \frac{\pi}{160}$)")
    plt.legend()
    plt.savefig("./outputs/comparison_graph.png", dpi=300)
    plt.clf() 
    
    # ------------------------------------------
    # (c) plot log-log graph of L2 error vs h
    # ------------------------------------------
    plt.loglog(h_values, l2_values, marker="o")
    plt.xlabel(r"$h$")
    plt.ylabel(r"$e_{L^2}(h)$")
    plt.title(r"Log-Log Plot of $L^2$ Error vs Step Size $h$")
    plt.savefig("./outputs/error_graph.png", dpi=300)
    