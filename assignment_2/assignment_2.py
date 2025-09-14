# import math <-- not used atm
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1 / (1 + 20 * x**2)


def lagrange_coefficients(nodes):
    x = nodes
    num_nodes = len(nodes)
    dd_table = np.array([[f(x) for x in nodes]])

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


if __name__ == "__main__":
    x = np.linspace(-1, 1, 100)
    y = f(x)

    five_x = np.linspace(-1, 1, 5)
    five_y = calculate_lagrange_output(five_x)

    ten_x = np.linspace(-1, 1, 10)
    ten_y = calculate_lagrange_output(ten_x)

    twenty_x = np.linspace(-1, 1, 20)
    twenty_y = calculate_lagrange_output(twenty_x)

    plt.plot(x, y)
    plt.plot(five_x, five_y)
    plt.plot(ten_x, ten_y)
    plt.plot(twenty_x, twenty_y)
    plt.show()
