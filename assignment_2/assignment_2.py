import math
import numpy as np
import matplotlib.pyplot as plt


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


def calculate_lagrange_output(nodes):
    a = lagrange_coefficients(nodes)
    y_coordinates = np.array([calculate_lagrange(nodes, a, x) for x in nodes])

    return y_coordinates


def chebyshev_nodes(n):
    nodes = np.array(
        [(math.cos((2 * k - 1) * math.pi / (2 * n))) for k in range(1, n+1)]
    )

    return nodes

def plot_functions(x_sets, y_sets, color_nums, markers, labels, title="", xlabel=r"$x$", ylabel=r"$y$"):
    # Enable Latex
    plt.rcParams["text.usetex"] = True
    # Styling
    plt.style.use("seaborn-v0_8-deep")
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']
    
    total_functions = len(x_sets)
    fig, ax = plt.subplots()
    for i in range(total_functions):
        ax.plot(x_sets[i], y_sets[i], color=default_colors[color_nums[i]], marker=markers[i], label=labels[i])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig

if __name__ == "__main__":
    # Question 2
    x = np.linspace(-1, 1, 100)
    y = f(x)

    # Question 2.2 (Equidistant Nodes)
    # n+1 nodes for P_n
    five_x = np.linspace(-1, 1, 6)
    five_y = calculate_lagrange_output(five_x)

    ten_x = np.linspace(-1, 1, 11)
    ten_y = calculate_lagrange_output(ten_x)

    twenty_x = np.linspace(-1, 1, 21)
    twenty_y = calculate_lagrange_output(twenty_x)
    
    x_sets = [x, five_x, ten_x, twenty_x]
    y_sets = [y, five_y, ten_y, twenty_y]
    labels = [r"$f(x)$",r"$P_5(x)$",r"$P_{10}(x)$",r"$P_{20}(x)$"]
    title = 'Equidistant Lagrange Interpolation'
    
    fig1 = plot_functions(x_sets[:2], y_sets[:2], [0,1], ['','.'], labels[:2], title+" (n = 5)")
    fig2 = plot_functions(x_sets[:3:2], y_sets[:3:2], [0,2], ['','.'], labels[:3:2], title+" (n = 10)")
    fig3 = plot_functions(x_sets[:4:3], y_sets[:4:3], [0,3], ['','.'],labels[:4:3], title+" (n = 20)")
    fig4 = plot_functions(x_sets, y_sets, [i for i in range(4)], ['','','',''], labels, title)
    
    # Save Figures
    # fig1.savefig('<path>/plots_2/p5.png', dpi = 300)
    # fig2.savefig('<path>/plots_2/p10.png', dpi = 300)
    # fig3.savefig('<path>/plots_2/p20.png', dpi = 300)
    # fig4.savefig('<path>/plots_2/all.png', dpi = 300)
    
    plt.show()
    
    # Question 2.3 (Chebyshev Nodes)
    five_x = chebyshev_nodes(6)
    five_y = calculate_lagrange_output(five_x)

    ten_x = chebyshev_nodes(11)
    ten_y = calculate_lagrange_output(ten_x)

    twenty_x = chebyshev_nodes(21)
    twenty_y = calculate_lagrange_output(twenty_x)
    
    x_sets = [x, five_x, ten_x, twenty_x]
    y_sets = [y, five_y, ten_y, twenty_y]
    labels = [r"$f(x)$",r"$P_5(x)$",r"$P_{10}(x)$",r"$P_{20}(x)$"]
    title = 'Chebyshev Lagrange Interpolation'
    
    fig1 = plot_functions(x_sets[:2], y_sets[:2], [0,1], ['','.'], labels[:2], title+" (n = 5)")
    fig2 = plot_functions(x_sets[:3:2], y_sets[:3:2], [0,2], ['','.'], labels[:3:2], title+" (n = 10)")
    fig3 = plot_functions(x_sets[:4:3], y_sets[:4:3], [0,3], ['','.'],labels[:4:3], title+" (n = 20)")
    fig4 = plot_functions(x_sets, y_sets, [i for i in range(4)], ['','','',''], labels, title)
    
    # Save Figures
    # fig1.savefig('<path>/plots_2/q2_3/chebyshev_p5.png', dpi = 300)
    # fig2.savefig('<path>/plots_2/q2_3/chebyshev_p10.png', dpi = 300)
    # fig3.savefig('<path>/plots_2/q2_3/chebyshev_p20.png', dpi = 300)
    # fig4.savefig('<path>/plots_2/q2_3/chebyshev_all.png', dpi = 300)
    
    plt.show()
