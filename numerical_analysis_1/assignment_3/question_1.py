import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + 25 * x**2)


def f_prime(x):
    return -50 * x / (1 + 25 * x**2) ** 2


def h_1(x):
    return (-625 / 338) * x**3 - (475 / 169) * x**2 + 1


def h_2(x):
    return (625 / 338) * x**3 - (475 / 169) * x**2 + 1


def hermite_coefficients(nodes, f=f, f_prime=f_prime):
    # Parameterize nodes for Hermite interpolation
    z = np.concatenate((nodes, nodes))
    sorted_indexes = z.argsort()
    z = z[sorted_indexes]
    num_nodes = len(z)

    # Set up dd table with zeroth dd
    dd_table = np.array([[f(zi) for zi in z]])

    # First divided difference
    f_prime_nodes = np.array([f_prime(xi) for xi in nodes])
    zeros = np.zeros(len(nodes))
    first_dd = np.concatenate((f_prime_nodes, zeros))[sorted_indexes]
    for j in range(num_nodes - 1):
        if j % 2 == 1:
            first_dd[j] = (dd_table[0, j + 1] - dd_table[0, j]) / (z[j + 1] - z[j])
    dd_table = np.vstack([dd_table, first_dd])

    # Remaining Divided Differences
    for i in range(2, num_nodes):
        ith_dd = np.zeros(num_nodes)

        for j in range(num_nodes - i):
            # Calculate ith divided differences
            ith_dd[j] = (dd_table[i - 1, j + 1] - dd_table[i - 1, j]) / (
                z[j + i] - z[j]
            )

        dd_table = np.vstack([dd_table, ith_dd])

    coefficients = np.array([dd_table[i, 0] for i in range(dd_table.shape[0])])
    return coefficients, dd_table.T, z


def generate_hermite(nodes, n):
    # Get coefficients
    a = hermite_coefficients(nodes)[0]
    # Start with function and constant term
    equation = f"P_{{{2 * n + 1}}}(x) = {a[0]}"
    w = []
    # Build (x - xi) terms
    for xi in nodes:
        if abs(xi) <= 1e-14:
            w.append("(x)")
        elif xi < 0:
            w.append(f"(x + {abs(xi)})")
        else:
            w.append(f"(x - {xi})")
    b = ""

    # Build polynomial string
    for i in range(1, len(a)):
        for j in range(i):
            # Multiply (x - xi) terms
            b += "^2" if j % 2 == 1 else w[int(j / 2)]
        # Multiply (x - xi) product with current coefficient and add term
        if a[i] > 0:
            equation += f" + {a[i]}{b}"
        elif a[i] < 0:
            equation += f" - {abs(a[i])}{b}"
        b = ""

    return equation


def calculate_hermite(nodes, x_coords=[], function=f):
    # Get coefficients
    a, z = hermite_coefficients(nodes, function)[:3:2]
    # If no x_coords provided, use nodes as x_coords
    if len(x_coords) == 0:
        x_coords = nodes
    # Start with constant term
    y = a[0]
    # Build (x - zi) terms
    w = [(x_coords - zi) for zi in z]
    # Temporary variable to hold (x - zi) product
    b = 1
    for i in range(1, len(a)):
        for j in range(i):
            # Multiply (x - zi) terms
            b *= w[j]
        # Multiply (x - zi) product with current coefficient
        y += a[i] * b
        b = 1

    return y


if __name__ == "__main__":
    nodes_1 = np.array([-1, 0])
    equation_1 = generate_hermite(nodes_1, 1)
    table_1, z_1 = hermite_coefficients(nodes_1)[1:3]

    nodes_2 = np.array([0, 1])
    equation_2 = generate_hermite(nodes_2, 1)
    table_2, z_2 = hermite_coefficients(nodes_2)[1:3]

    equations = [equation_1, equation_2]
    tables = [table_1, table_2]
    z = [z_1, z_2]

    with open("./outputs_3/hermite.txt", "w") as file:
        for k in range(2):
            file.write(equations[k] + "\n\n")

            file.write("\\begin{center}\n")
            file.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
            file.write("\\hline\n")
            file.write("& $z_i$ & $f[z_i]$ & 1st dd. & 2nd dd. & 3rd dd. \\\\\n")
            file.write("\\hline\n")
            for i in range(tables[k].shape[0]):
                file.write(f"$z_{i}$ & ${z[k][i]}$ ")
                for j in range(tables[k].shape[1]):
                    file.write(f"& ${tables[k][i, j]:.4f}$ ")
                file.write("\\\\ \n")
            file.write("\\hline\n")
            file.write("\\end{tabular}\n")
            file.write("\\end{center}\n\n")

    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    
    x = np.linspace(-1, 1, 200)
    f_y = f(x)

    conditions = [x <= 0, x > 0]
    h_y = np.piecewise(x, conditions, [h_1, h_2])
    p_y = np.concatenate(
        (calculate_hermite(nodes_1, x[:100]), calculate_hermite(nodes_2, x[100:]))
    )
    
    plt.plot(x, f_y, label="$f(x)$")
    plt.plot(x, h_y, label="$H(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.title("Function and Cubic Hermite")
    plt.savefig("./outputs_3/hermite_plot_h.png", dpi=300)
    plt.cla()
    
    plt.plot(x, f_y, label="$f(x)$")
    plt.plot(x, p_y, label="$P(x)$", color = default_colors[2])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.title("Function and Cubic Hermite (Newton Form)")
    plt.savefig("./outputs_3/hermite_plot_p.png", dpi=300)
    plt.cla()
    
    plt.plot(x, f_y, label="$f(x)$")
    plt.plot(x, h_y, label="$H(x)$")
    plt.plot(x, p_y, label="$P(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.title("Function and Cubic Hermite Interpolations")
    plt.savefig("./outputs_3/hermite_plot.png", dpi=300)