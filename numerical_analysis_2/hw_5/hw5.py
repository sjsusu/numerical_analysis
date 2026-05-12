import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Space discretization
N=6
dx=1/N
X_NODES = np.linspace(0, 1, N+1)
# Time discretization
M=4
dt = 1/M
T_NODES = np.linspace(0, 1, M+1)
# Triangular Element area
AREA = (dx*dt)/2

def u_exact(x,t):
    return np.sin(np.pi*x)*np.cos(t)

def f(x,t):
    return (np.pi**2 -1)*np.sin(np.pi*x)*np.cos(t)

def generate_global_index_matrix():
    '''
    Generates a matrix of global indices for the nodes in the mesh as helper for generate_connectivity_matrix_and_nodal_values method.
    '''
    
    global_index_matrix = np.zeros((M+1, N+1), dtype=int)
    count = 0
    for i in range(M, -1, -1):
        for j in range(N+1):
            global_index_matrix[i, j] = count
            count += 1
    return global_index_matrix

def generate_connectivity_matrix_and_nodal_values(global_index_matrix):
    '''
    Generates the connectivity matrix and nodal values for the elements in the mesh.
    
    The nodal values have the format [[x1, t1], [x2, t2], [x3, t3]] for each element, where (x1, t1), (x2, t2), and (x3, t3) are the coordinates of the three nodes of the triangle. The connectivity matrix has the format [[n1, n2, n3], ...] where n1, n2, and n3 are the global indices of the nodes in the triangle.
    '''
    
    connectivity_matrix = []
    nodal_values=[]
    for i in range(M, 0, -1):
        for j in range(N):
            bottom_left = global_index_matrix[i, j]
            bottom_right = global_index_matrix[i, j+1]
            top_right = global_index_matrix[i-1, j+1]
            top_left = global_index_matrix[i-1, j]
            connectivity_matrix.append([bottom_left, bottom_right, top_right])
            connectivity_matrix.append([bottom_left, top_right, top_left])
            
            bottom_left_values = [X_NODES[j], T_NODES[i]]
            bottom_right_values = [X_NODES[j+1], T_NODES[i]]
            top_right_values = [X_NODES[j+1], T_NODES[i-1]]
            top_left_values = [X_NODES[j], T_NODES[i-1]]
            nodal_values.append([bottom_left_values, bottom_right_values, top_right_values])
            nodal_values.append([bottom_left_values, top_right_values, top_left_values])
    return np.array(connectivity_matrix), np.array(nodal_values)

def generate_element_basis_coefficients(nodal_values):
    '''
    Given an element's nodal values, generates the coefficients of the basis functions for that element.
    '''
    
    coefficients = []
    for i in range(3):
        # ex. c1 = (x1-x2)*(t1-t3) - (x1-x3)*(t1-t2)
        ci = (nodal_values[i][0]-nodal_values[(i+1)%3][0])*(nodal_values[i][1]-nodal_values[(i+2)%3][1]) - (nodal_values[i][0]-nodal_values[(i+2)%3][0])*(nodal_values[i][1]-nodal_values[(i+1)%3][1])
        
        # ex. a1 = (1/c1) * (x2t3-x3t2) 
        ai = (1/ci) * (nodal_values[(i+1)%3][0]*nodal_values[(i+2)%3][1] - nodal_values[(i+2)%3][0]*nodal_values[(i+1)%3][1])
        
        # ex. b1 = (1/c1) * (t3-t2) 
        bi = (1/ci) * (nodal_values[(i+2)%3][1] - nodal_values[(i+1)%3][1])
        
        # ex. d1 = (1/c1) * (x2-x3)
        di = (1/ci) * (nodal_values[(i+1)%3][0] - nodal_values[(i+2)%3][0])
        coefficients.append([ai, bi, di])

    return np.array(coefficients)

def generate_element_stiffness_and_force(element_nodal_values):
    element_stiffness = np.zeros((3, 3))
    element_force = np.zeros(3)
    
    basis_coefficients = generate_element_basis_coefficients(element_nodal_values)
    b = basis_coefficients[:, 1]
    d = basis_coefficients[:, 2]
    
    for i in range(3):
        for j in range(3):
            # A_ij = (bj*bi + dj*di) * area
            A_ij = (b[j]*b[i] - d[j]*d[i]) * AREA
            element_stiffness[i, j] = A_ij
            
        # Quadrature for the force term, using the centroid of the triangle as the quadrature point
        # Fi = (area / 3) * f(xc, tc), where (xc, tc) is the centroid of the triangle
        xc = np.mean(element_nodal_values[:, 0])
        tc = np.mean(element_nodal_values[:, 1])
        Fi = (AREA / 3) * f(xc, tc) 
        element_force[i] = Fi
    return element_stiffness, element_force

def assemble_global_system(connectivity_matrix, nodal_values):
    num_nodes = (N+1)*(M+1)
    global_stiffness = np.zeros((num_nodes, num_nodes))
    global_force = np.zeros(num_nodes)
    
    for element_index in range(connectivity_matrix.shape[0]):
        element_nodal_values = nodal_values[element_index]
        element_stiffness, element_force = generate_element_stiffness_and_force(element_nodal_values)
        
        for i in range(3):
            global_i = connectivity_matrix[element_index, i]
            global_force[global_i] += element_force[i]
            for j in range(3):
                global_j = connectivity_matrix[element_index, j]
                global_stiffness[global_i, global_j] += element_stiffness[i, j]
    
    return global_stiffness, global_force

def fem_solver():
    global_index_matrix = generate_global_index_matrix()
    connectivity_matrix, nodal_values = generate_connectivity_matrix_and_nodal_values(global_index_matrix)
    global_stiffness, global_force = assemble_global_system(connectivity_matrix, nodal_values)
    
    # Apply boundary conditions (u(x,0)=sin(pi*x), u(0,t)=u(1,t)=0)
    # Initial condition (t = 0)
    for j in range(N+1):
        k = global_index_matrix[M, j]
        # Zero out row
        global_stiffness[k, :] = 0
        # Set diagonal to 1
        global_stiffness[k, k] = 1
        # Set force to sin(pi*x)
        global_force[k] = np.sin(np.pi*X_NODES[j])

    # Spatial boundaries (x = 0 and x = 1)
    for i in range(M+1):
        # left boundary x=0
        k = global_index_matrix[i, 0]
        global_stiffness[k, :] = 0
        global_stiffness[k, k] = 1
        global_force[k] = 0

        # right boundary x=1
        k = global_index_matrix[i, N]
        global_stiffness[k, :] = 0
        global_stiffness[k, k] = 1
        global_force[k] = 0
    
    # Solve the linear system
    solution = solve(global_stiffness, global_force)
    
    # Reshape the solution into 2D array matching global index matrix
    solution_grid = np.zeros((M+1, N+1))
    for i in range(M+1):
        for j in range(N+1):
            k = global_index_matrix[i, j]
            solution_grid[i, j] = solution[k]
    
    return solution_grid

def plot_solution(solution_grid, name):
    # plot solution in 3D
    X = np.zeros((M+1, N+1))
    T = np.zeros((M+1, N+1))
    for i in range(M, -1, -1):
        for j in range(N+1):
            X[i, j] = X_NODES[j]
            T[i, j] = T_NODES[M-i]
            
    u_exact_grid = u_exact(X, T)
    fig, ax = plt.subplots(1,2, subplot_kw={"projection": "3d"}, figsize=(12, 7))
    ax[0].plot_surface(X, T, solution_grid, cmap='viridis')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].set_zlabel('u(x,t)')
    ax[0].set_title('FEM Solution')
    
    ax[1].plot_surface(X, T, u_exact_grid, cmap='viridis')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    ax[1].set_zlabel('u(x,t)')
    ax[1].set_title('Exact Solution')
    plt.tight_layout()
    plt.savefig(name ,dpi=300)
    
def compute_max_error(solution_grid):
    X = np.zeros((M+1, N+1))
    T = np.zeros((M+1, N+1))
    for i in range(M, -1, -1):
        for j in range(N+1):
            X[i, j] = X_NODES[j]
            T[i, j] = T_NODES[M-i]
    u_exact_grid = u_exact(X, T)
    error_grid = np.abs(solution_grid - u_exact_grid)
    max_error = np.max(error_grid)
    print("Max Error:", max_error)

if __name__ == "__main__":
    solution = fem_solver()
    plot_solution(solution, "N6_M4.png")
    compute_max_error(solution)