import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from figures import Figure


def f(x):
    return 1 / (1 + 20 * x**2)


def lagrange_coefficients(nodes):
    x = nodes
    num_nodes = len(nodes)
    dd_table = np.array([[f(xi) for xi in nodes]])

    for i in range(1, num_nodes):
        ith_dd = np.zeros(num_nodes)

        for j in range(num_nodes - i):
            ith_dd[j] = (dd_table[i - 1, j + 1] - dd_table[i - 1, j]) / (
                x[j + i] - x[j]
            )

        dd_table = np.vstack([dd_table, ith_dd])

    a = np.array([dd_table[i, 0] for i in range(dd_table.shape[0])])
    return a


def calculate_lagrange(nodes, a, x):
    y = a[0]
    w = [(x - xi) for xi in nodes]
    b = 1

    for i in range(1, len(a)):
        for j in range(i):
            b *= w[j]
        y += a[i] * b
        b = 1

    return y


def generate_lagrange(nodes, degree):
    a = lagrange_coefficients(nodes)
    equation = f"P_{{{degree}}}(x) = {a[0]}"
    w = []
    for xi in nodes:
        if abs(xi) <= 1e-14:
            w.append(f"({xi})")
        elif xi < 0:
            w.append(f"(x + {abs(xi)})")
        else:
            w.append(f"(x - {xi})")
    b = ""

    for i in range(1, len(a)):
        for j in range(i):
            b += w[j]
        if a[i] >= 0:
            equation += f" + {a[i]}{b}"
        else:
            equation += f" - {abs(a[i])}{b}"
        b = ""

    return equation


def calculate_lagrange_output(nodes, x_coords=[]):
    a = lagrange_coefficients(nodes)
    if len(x_coords) == 0:
        x_coords = nodes
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
    return equation


if __name__ == "__main__":
    # Question 2
    title = "Equidistant Lagrange Interpolation"
    x = np.linspace(-1, 1, 100)
    y = f(x)
    fig_f = Figure(x, y, r"$f(x)$")

    # Question 2.2 (Equidistant Nodes)
    # -----------------------------------------------------------
    nodes_deg_5 = np.linspace(-1, 1, 6)
    y_nodes_5 = calculate_lagrange_output(nodes_deg_5)
    y_poly_5 = calculate_lagrange_output(nodes_deg_5, x)
    f1 = Figure(x, y_poly_5, r"$P_5(x)$", 1)
    f2 = Figure(nodes_deg_5, y_nodes_5, "", 1, ".", "")

    nodes_deg_10 = np.linspace(-1, 1, 11)
    y_nodes_10 = calculate_lagrange_output(nodes_deg_10)
    y_poly_10 = calculate_lagrange_output(nodes_deg_10, x)
    f3 = Figure(x, y_poly_10, r"$P_{10}(x)$", 2)
    f4 = Figure(nodes_deg_10, y_nodes_10, "", 2, ".", "")

    nodes_deg_20 = np.linspace(-1, 1, 21)
    y_nodes_20 = calculate_lagrange_output(nodes_deg_20)
    y_poly_20 = calculate_lagrange_output(nodes_deg_20, x)
    f5 = Figure(x, y_poly_20, r"$P_{20}(x)$", 3)
    f6 = Figure(nodes_deg_20, y_nodes_20, "", 3, ".", "")

    # Merge and plot figures
    fig1 = fig_f.copy().merge([f1, f2], title + " (n = 5)")
    fig2 = fig_f.copy().merge([f3, f4], title + " (n = 10)")
    fig3 = fig_f.copy().merge([f5, f6], title + " (n = 20)")
    fig4 = fig_f.copy().merge([f1, f3, f5])
    fig1.get_figure("./plots_2/q2_2/p5.png")
    fig2.get_figure("./plots_2/q2_2/p10.png")
    fig3.get_figure("./plots_2/q2_2/p20.png")
    fig4.get_figure("./plots_2/q2_2/all.png")

    # Generate and save equations to a text file
    equations = [
        generate_lagrange(nodes_deg_5, 5),
        generate_lagrange(nodes_deg_10, 10),
        generate_lagrange(nodes_deg_20, 20),
    ]

    with open("./plots_2/q2_2/lagrange_equations.txt", "w") as file:
        for eq in equations:
            file.write(eq + "\n")

    plt.show()
    # -----------------------------------------------------------

    # Question 2.3 (Chebyshev Nodes)
    # -----------------------------------------------------------
    title = "Chebyshev Lagrange Interpolation"

    nodes_deg_5 = chebyshev_nodes(6)
    y_nodes_5 = calculate_lagrange_output(nodes_deg_5)
    y_poly_5 = calculate_lagrange_output(nodes_deg_5, x)
    f1 = Figure(x, y_poly_5, r"$P_5(x)$", 1)
    f2 = Figure(nodes_deg_5, y_nodes_5, "", 1, ".", "")

    nodes_deg_10 = chebyshev_nodes(11)
    y_nodes_10 = calculate_lagrange_output(nodes_deg_10)
    y_poly_10 = calculate_lagrange_output(nodes_deg_10, x)
    f3 = Figure(x, y_poly_10, r"$P_{10}(x)$", 2)
    f4 = Figure(nodes_deg_10, y_nodes_10, "", 2, ".", "")

    nodes_deg_20 = chebyshev_nodes(21)
    y_nodes_20 = calculate_lagrange_output(nodes_deg_20)
    y_poly_20 = calculate_lagrange_output(nodes_deg_20, x)
    f5 = Figure(x, y_poly_20, r"$P_{20}(x)$", 3)
    f6 = Figure(nodes_deg_20, y_nodes_20, "", 3, ".", "")

    # Merge and plot figures
    fig1 = fig_f.copy().merge([f1, f2], title + " (n = 5)")
    fig2 = fig_f.copy().merge([f3, f4], title + " (n = 10)")
    fig3 = fig_f.copy().merge([f5, f6], title + " (n = 20)")
    fig4 = fig_f.copy().merge([f1, f3, f5])
    fig1.get_figure("./plots_2/q2_3/chebyshev_p5.png")
    fig2.get_figure("./plots_2/q2_3/chebyshev_p10.png")
    fig3.get_figure("./plots_2/q2_3/chebyshev_p20.png")
    fig4.get_figure("./plots_2/q2_3/chebyshev_all.png")

    # Generate and save equations to a text file
    equations = [
        generate_lagrange(nodes_deg_5, 5),
        generate_lagrange(nodes_deg_10, 10),
        generate_lagrange(nodes_deg_20, 20),
    ]

    with open("./plots_2/q2_3/lagrange_equations_chebyshev.txt", "w") as file:
        for eq in equations:
            file.write(eq + "\n")

    plt.show()
    # -----------------------------------------------------------

    # Question 3.3 (Chebyshev Polynomials)
    # -----------------------------------------------------------
    with open("./plots_2/q3_3/chebyshev_polynomials.txt", "w") as file:
        for n in range(11):
            equation = generate_chebyshev(n)
            file.write(equation + "\n")
    # -----------------------------------------------------------
