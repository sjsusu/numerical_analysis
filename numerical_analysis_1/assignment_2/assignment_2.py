import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as smp


def f(x):
    return 1 / (1 + 20 * x**2)


def lagrange_coefficients(nodes, function=f):
    x = nodes
    num_nodes = len(nodes)
    # Add zeroth divided differences
    dd_table = np.array([[function(xi) for xi in nodes]])

    # Calculate divided difference table
    for i in range(1, num_nodes):
        ith_dd = np.zeros(num_nodes)

        for j in range(num_nodes - i):
            # Calculate ith divided differences
            ith_dd[j] = (dd_table[i - 1, j + 1] - dd_table[i - 1, j]) / (
                x[j + i] - x[j]
            )

        # Append the ith divided difference row to the table
        dd_table = np.vstack([dd_table, ith_dd])

    # Extract coefficients (first column of the table)
    a = np.array([dd_table[i, 0] for i in range(dd_table.shape[0])])
    return a


def calculate_lagrange(nodes, a, x):
    # Start with constant term
    y = a[0]
    # Build (x - xi) terms
    w = [(x - xi) for xi in nodes]
    # Temporary variable to hold (x - xi) product
    b = 1

    for i in range(1, len(a)):
        for j in range(i):
            # Multiply (x - xi) terms
            b *= w[j]
        # Multiply (x - xi) product with current coefficient
        y += a[i] * b
        b = 1

    return y


def generate_lagrange(nodes, degree):
    # Get coefficients
    a = lagrange_coefficients(nodes)
    # Start with function and constant term
    equation = f"P_{{{degree}}}(x) = {a[0]}"
    w = []
    # Build (x - xi) terms
    for xi in nodes:
        if abs(xi) <= 1e-14:
            w.append('(x)')
        elif xi < 0:
            w.append(f"(x + {abs(xi)})")
        else:
            w.append(f"(x - {xi})")
    b = ""

    # Build polynomial string
    for i in range(1, len(a)):
        for j in range(i):
            # Multiply (x - xi) terms
            b += w[j]
        # Multiply (x - xi) product with current coefficient and add term
        if a[i] >= 0:
            equation += f" + {a[i]}{b}"
        else:
            equation += f" - {abs(a[i])}{b}"
        b = ""

    return equation


def calculate_lagrange_output(nodes, x_coords=[], function=f):
    # Get coefficients
    a = lagrange_coefficients(nodes, function)
    # If no x_coords provided, use nodes as x_coords
    if len(x_coords) == 0:
        x_coords = nodes
    # Calculate y coordinates for each x coordinate
    y_coordinates = np.array([calculate_lagrange(nodes, a, x) for x in x_coords])

    return y_coordinates


def chebyshev_nodes(n):
    nodes = np.array(
        [(math.cos((2 * k - 1) * math.pi / (2 * n))) for k in range(1, n + 1)]
    )

    return nodes


def chebyshev_poly(n):
    # Base Cases
    if n == 0:
        return 1
    elif n == 1:
        return smp.symbols("x")
    # Recursive Case
    else:
        return 2 * smp.symbols("x") * chebyshev_poly(n - 1) - chebyshev_poly(n - 2)


def generate_chebyshev(n):
    x = smp.symbols("x")
    poly = smp.expand(chebyshev_poly(n))
    function = f"T_{{{n}}}({x})"
    equation = function + " = " + smp.latex(poly)
    return equation, poly


# Main Method
if __name__ == "__main__":
    # Enable Latex and Styling
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")
    # Get list of default colors for style
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]

    # Question 2
    # -----------------------------------------------------------
    # Question 2.1 (Find P5(x) Equation)
    # -----------------------------------------------------------
    # Generate and save equation to a text file
    nodes = np.linspace(-1, 1, 6)
    equation = generate_lagrange(nodes, 5)
    with open("./plots_2/q2_1/p5.txt", "w") as file:
            file.write(equation)
            
    # -----------------------------------------------------------
    # Question 2.2 (Equidistant Nodes)
    # -----------------------------------------------------------
    title = "Equidistant Lagrange Interpolation"
    x_coords = np.linspace(-1, 1, 100)
    y_coords = f(x_coords)
    n = [5, 10, 20]

    for i in range(len(n) + 1):
        fig, ax = plt.subplots()
        ax.plot(x_coords, y_coords, label=r"$f(x)$")

        # Plot f(x), P(x), and nodes
        if i <= len(n) - 1:
            x_nodes = np.linspace(-1, 1, n[i] + 1)
            y_nodes = calculate_lagrange_output(x_nodes)
            y_poly = calculate_lagrange_output(x_nodes, x_coords)
            ax.plot(
                x_coords,
                y_poly,
                label=rf"$P_{{{n[i]}}}(x)$",
                color=default_colors[i + 1],
            )
            ax.scatter(x_nodes, y_nodes, color=default_colors[i + 1])
            ax.set_title(title + f" (n = {n[i]})")
            path = f"./plots_2/q2_2/p{n[i]}.png"

        # Plot f(x) with all P(x)
        else:
            for j in range(len(n)):
                x_nodes = np.linspace(-1, 1, n[j] + 1)
                y_nodes = calculate_lagrange_output(x_nodes)
                y_poly = calculate_lagrange_output(x_nodes, x_coords)
                ax.plot(
                    x_coords,
                    y_poly,
                    label=rf"$P_{{{n[j]}}}(x)$",
                    color=default_colors[j + 1],
                )
                ax.set_title(title)
                path = "./plots_2/q2_2/all.png"

        # Configure axis and save figure
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.legend()
        fig.savefig(path, dpi=300)
        plt.show()
        ax.cla()

    # -----------------------------------------------------------
    # Question 2.3 (Chebyshev Nodes)
    # -----------------------------------------------------------
    title = "Chebyshev Lagrange Interpolation"
    x_coords = np.linspace(-1, 1, 100)
    y_coords = f(x_coords)
    n = [5, 10, 20]

    for i in range(len(n) + 1):
        fig, ax = plt.subplots()
        ax.plot(x_coords, y_coords, label=r"$f(x)$")

        # Plot f(x), P(x), and nodes
        if i <= len(n) - 1:
            x_nodes = chebyshev_nodes(n[i] + 1)
            y_nodes = calculate_lagrange_output(x_nodes)
            y_poly = calculate_lagrange_output(x_nodes, x_coords)
            ax.plot(
                x_coords,
                y_poly,
                label=rf"$P_{{{n[i]}}}(x)$",
                color=default_colors[i + 1],
            )
            ax.scatter(x_nodes, y_nodes, color=default_colors[i + 1])
            ax.set_title(title + f" (n = {n[i]})")
            path = f"./plots_2/q2_3/chebyshev_p{n[i]}.png"

        # Plot f(x) with all P(x)
        else:
            for j in range(len(n)):
                x_nodes = chebyshev_nodes(n[j] + 1)
                y_nodes = calculate_lagrange_output(x_nodes)
                y_poly = calculate_lagrange_output(x_nodes, x_coords)
                ax.plot(
                    x_coords,
                    y_poly,
                    label=rf"$P_{{{n[j]}}}(x)$",
                    color=default_colors[j + 1],
                )
                ax.set_title(title)
                path = "./plots_2/q2_3/chebyshev_all.png"

        # Configure axis and save figure
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.legend()
        fig.savefig(path, dpi=300)
        plt.show()
        ax.cla()
        
    # -----------------------------------------------------------
    # Question 3 
    # -----------------------------------------------------------
    # Question 3.3 (Chebyshev Polynomials)
    # -----------------------------------------------------------
    # Polynomials list used for plotting in 3.4
    polys = []

    # Generate and save equations to a text file
    with open("./plots_2/q3_3/chebyshev_polynomials.txt", "w") as file:
        for n in range(6):
            equation, poly = generate_chebyshev(n)
            polys.append(poly)
            file.write(equation + "\n")

    # -----------------------------------------------------------
    # Question 3.34 (Plot Chebyshev Polynomials)
    # -----------------------------------------------------------
    # Plot Chebyshev Polynomials
    x_coords = np.linspace(-1, 1, 100)
    x = smp.symbols("x")

    fig, ax = plt.subplots()
    # Plot each polynomial
    for i in range(6):
        # Evaluate polynomial at each x coordinate
        if i != 0:
            y_coords = np.array(
                [polys[i].evalf(subs={x: x_coords[j]}) for j in range(len(x_coords))]
            )
        # Case for T_0(x) = 1 
        else:
            y_coords = np.array([1 for _ in range(len(x_coords))])
        ax.plot(x_coords, y_coords, label=rf"$T_{{{i}}}x)$")

    # Configure axis and save figure
    ax.set_title("Chebyshev Polynomials")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()
    fig.savefig("./plots_2/q3_4/chebyshev_polys.png", dpi=300)
    plt.show()

    # -----------------------------------------------------------
    # Question 4
    # -----------------------------------------------------------
    # Question 4.3 (Lagrange for Nonsmooth Functions)
    # -----------------------------------------------------------
    title = "Equidistant Lagrange Interpolation"
    x_coords = np.linspace(-1, 1, 200)
    y_coords = abs(x_coords)
    n = [2, 4, 6, 8]

    # Plot f(x), P(x), nodes, and |R(x)| for each n
    for i in range(len(n)+1):
        fig, ax = plt.subplots()

        if i < len(n):
            ax.plot(x_coords, y_coords, label=r"$f(x)$")
            x_nodes = np.linspace(-1, 1, n[i] + 1)
            y_nodes = calculate_lagrange_output(x_nodes, x_nodes, abs)
            y_poly = calculate_lagrange_output(x_nodes, x_coords, abs)
            y_error = np.abs(y_coords - y_poly)
            ax.plot(
                x_coords,
                y_poly,
                label=rf"$P_{{{n[i]}}}(x)$",
                color=default_colors[i + 1],
            )
            ax.scatter(x_nodes, y_nodes, color=default_colors[i + 1])
            ax.plot(
                x_coords,
                y_error,
                label=rf"$|R_{{{n[i]}}}(x)|$",
                linestyle="dashed",
                color='#e82351'
            )
            ax.set_title(title + f" (n = {n[i]})")
            path = f"./plots_2/q4_3/p{n[i]}.png"
        else:
            # Plot |R(x)| for all n
            for i in range(len(n)):
                x_nodes = np.linspace(-1, 1, n[i] + 1)
                y_nodes = calculate_lagrange_output(x_nodes, x_nodes, abs)
                y_poly = calculate_lagrange_output(x_nodes, x_coords, abs)
                y_error = np.abs(y_coords - y_poly)
                ax.plot(
                    x_coords,
                    y_error,
                    label=rf"$|R_{{{n[i]}}}(x)|$",
                    color=default_colors[i]
                )
            path = "./plots_2/q4_3/error.png"
            ax.set_title(title + " Error")
            

        # Configure axis and save figure
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.legend()
        fig.savefig(path, dpi=300)
        plt.show()
        ax.cla()
    # -----------------------------------------------------------
