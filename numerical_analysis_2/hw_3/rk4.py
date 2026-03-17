import numpy as np

def RK4_step(F, y, h):
    k1 = F(y)
    k2 = F(y + (h / 2) * k1)
    k3 = F(y + (h / 2) * k2)
    k4 = F(y + h * k3)

    y_new = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_new

def RK4_solver(F, y_0, h=0.01, T=np.pi/2):
    num_steps = int(T / h)
    # Adjust h to ensure we end exactly at T
    h = T / num_steps 
    y_values = np.zeros((num_steps + 1, len(y_0)))
    y_values[0] = y_0

    for n in range(num_steps):
        y_values[n + 1] = RK4_step(F, y_values[n], h)

    return y_values