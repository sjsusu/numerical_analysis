import numpy as np
import matplotlib.pyplot as plt

def forward_diff(f, x, step=1):
    return (f(x + step) - f(x)) / step

def three_point_forward_diff(f, x, step=1):
    return (-3 * f(x) + 4 * f(x + step) - f(x + 2 * step)) / (2 * step)

def calc_absolute_error(f, step, eval_x=1, diff_method=forward_diff):
    approx = diff_method(f, eval_x, step)
    return np.abs(approx - f(eval_x))

if __name__ == "__main__":
    with open("./outputs_3/forward_diff.txt", "w") as file:
        for k in range(1,11):
            h =10**(-k)
            f_prime = forward_diff(np.exp, 1.5, h)
            file.write(f'\\[D_h^{{({k})}}f(1.5) = {f_prime}\\]\n')

    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = True
    plt.rc("grid", color="#a6a6a6", linestyle="dotted", linewidth=0.5)
    plt.style.use("seaborn-v0_8-deep")
    
    h = np.logspace(-1, -10, 200)
    error = calc_absolute_error(np.exp, h)
    
    plt.loglog(h, error, label='$\\log|D_hf(1) - f\'(1)|$')
    plt.xlabel('$\\log(h)$')
    plt.title('Absolute Error with Forward Difference')
    plt.legend()
    plt.savefig('./outputs_3/forward_diff_graph.png', dpi=300)
    plt.show()
    
    with open("./outputs_3/three_point_forward_diff.txt", "w") as file:
        for k in range(1,11):
            h =10**(-k)
            f_prime = three_point_forward_diff(np.exp, 1.5, h)
            file.write(f'\\[D_h^{{({k})}}f(1.5) = {f_prime}\\]\n')
    
    h = np.logspace(-1, -10, 200)
    error = calc_absolute_error(np.exp, h, diff_method=three_point_forward_diff)
    
    plt.loglog(h, error, label='$\\log|D_hf(1) - f\'(1)|$')
    plt.xlabel('$\\log(h)$')
    plt.title('Absolute Error with Three Point Forward Difference')
    plt.legend()
    plt.savefig('./outputs_3/three_point_forward_diff_graph.png', dpi=300)
    plt.show()
    