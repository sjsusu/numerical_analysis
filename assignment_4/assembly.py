import numpy as np
import sympy as sp

def global_indexing(width, height=None):
    if height is None:
        height = width
    return np.arange(width * height).reshape((height, width))

def generate_connectivity_matrix(global_indices):
    total_elements = (global_indices.shape[0] - 1) * (global_indices.shape[1] - 1)
    connectivity_matrix = np.zeros((total_elements, 4), dtype=int)
    element = 0
    for i in range(global_indices.shape[0] - 1):
        for j in range(global_indices.shape[1] - 1):
            connectivity_matrix[element, 0] = global_indices[i+1, j]
            connectivity_matrix[element, 1] = global_indices[i+1, j+1]
            connectivity_matrix[element, 2] = global_indices[i, j+1]
            connectivity_matrix[element, 3] = global_indices[i, j]
            element += 1
    return connectivity_matrix

def element_mass_matrix(w, h):
    area = w * h
    Me = (area/36) * np.array([[4, 2, 1, 2],
                               [2, 4, 2, 1],
                               [1, 2, 4, 2],
                               [2, 1, 2, 4]])
    return Me

def element_stiffness_matrix(w, h):
    # When multiplying each entry formula by w * h /16, we get the following reduced formulas:
    wh = w/h
    hw = h/w
    # Main diagonal
    a = (1/3) * (hw + wh)
    # Skew diagonal
    b = (1/6) * (hw - 2*wh)
    # First off-diagonal
    c = -(1/6) * (2*hw - wh)
    # Second off-diagonal
    d = -(1/6) * (hw + wh)
    # Construct element stiffness matrix
    Ke = np.array([[a, c, d, b],
                   [c, a, b, d],
                   [d, b, a, c],
                   [b, d, c, a]])
    return Ke

def generate_global_coordinates(x_nodes, y_nodes=None):
    if y_nodes is None:
        y_nodes = x_nodes
        
    y_nodes = y_nodes[::-1]
    global_coordinates = []
    for y in y_nodes:
        for x in x_nodes:
            global_coordinates.append([x, y])
    global_coordinates = np.array(global_coordinates)
    return global_coordinates


def global_assembly(x_nodes, y_nodes=None):
    if y_nodes is None:
        y_nodes = x_nodes
        
    global_coordinates = generate_global_coordinates(x_nodes, y_nodes)
    
    width = len(x_nodes)
    height = len(y_nodes)

    global_indices = global_indexing(width, height)
    connectivity_matrix = generate_connectivity_matrix(global_indices)

    num_nodes = width * height
    M_global = np.zeros((num_nodes, num_nodes))
    K_global = np.zeros((num_nodes, num_nodes))
    
    for element in connectivity_matrix:
        
        # Bottom-left (node 0)
        x0 = global_coordinates[element[0], 0]
        y0 = global_coordinates[element[0], 1]

        # Bottom-right (node 1)
        x1 = global_coordinates[element[1], 0]

        # Top-left (node 3)
        y3 = global_coordinates[element[3], 1]

        # Element width and height
        w = abs(x1 - x0)
        h = abs(y3 - y0)

        # Get element matrices
        Me = element_mass_matrix(w, h)
        Ke = element_stiffness_matrix(w, h)

        # Add element contributions to global matrices
        for i_local in range(4):
            i_global = element[i_local]
            for j_local in range(4):
                j_global = element[j_local]
                M_global[i_global, j_global] += Me[i_local, j_local]
                K_global[i_global, j_global] += Ke[i_local, j_local]
    
    return global_coordinates, M_global, K_global

def classify_boundary_nodes(global_indexing):
    # Create a boundary indicator array where 1 indicates a boundary node
    boundary_indicator = np.zeros_like(global_indexing)
    boundary_indicator[0, :] = 1  # Top boundary
    boundary_indicator[-1, :] = 1  # Bottom boundary
    boundary_indicator[:, 0] = 1  # Left boundary
    boundary_indicator[:, -1] = 1  # Right boundary
    return boundary_indicator.flatten()

def implement_dirichlet_bc(M, K,boundary_indicator):
    num_nodes = M.shape[0]
    for i in range(num_nodes):
        if boundary_indicator[i] == 1:
            M[i, :] = 0
            M[:, i] = 0
            M[i, i] = 1
            K[i, :] = 0
            K[:, i] = 0
    return M, K

def u_0(coordinates, boundary_indicator):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    u = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    u[boundary_indicator == 1] = 0.0  # Apply Dirichlet BCs
    num_nodes = coordinates.shape[0]
    return u.reshape((num_nodes,1))

if __name__ == "__main__":
    width = 5  # Number of nodes along one dimension
    global_indices = global_indexing(width)
    connectivity_matrix = generate_connectivity_matrix(global_indices)
    
    x_nodes = np.array([-2, -1.6, -1.2, -0.8, 2])
    _, M, K = global_assembly(x_nodes)
    
    coordinates = generate_global_coordinates(x_nodes)
    boundary_indicator = classify_boundary_nodes(global_indices)
    u = u_0(coordinates, boundary_indicator).reshape(5,5)
    print(u)

    # avoid forming explicit inverse; use solve for M^{-1} K
    M_inv_K = np.linalg.solve(M, K)
    evals = np.linalg.eigvals(M_inv_K)
    max_eigenvalue = np.max(np.abs(evals))
    M_evals = np.linalg.eigvals(M)
    K_evals = np.linalg.eigvals(K)

    with open("./outputs_4/matrices.txt", "w") as f:
        latex_matrix = sp.latex(sp.Matrix(coordinates))
        f.write("Global Coordinates:\n")
        f.write(latex_matrix + "\n\n")
        
        # print(global_indices)
        latex_matrix = sp.latex(sp.Matrix(global_indices))
        f.write("Global Indices Matrix:\n")
        f.write(latex_matrix + "\n\n")
        
        # print(connectivity_matrix)
        latex_matrix = sp.latex(sp.Matrix(connectivity_matrix))
        f.write("Connectivity Matrix:\n")
        f.write(latex_matrix + "\n\n")

        M = np.round(M, 4)
        M_split = np.array_split(M, 3, axis=1)
        latex_1_matrix = sp.latex(sp.Matrix(M_split[0]))
        latex_2_matrix = sp.latex(sp.Matrix(M_split[1]))
        latex_3_matrix = sp.latex(sp.Matrix(M_split[2]))
        f.write("Mass Matrix (Part 1):\n")
        f.write(latex_1_matrix + "\n\n")
        f.write("Mass Matrix (Part 2):\n")
        f.write(latex_2_matrix + "\n\n")
        f.write("Mass Matrix (Part 3):\n")
        f.write(latex_3_matrix + "\n\n")

        K = np.round(K, 4)
        K_split = np.array_split(K, 3, axis=1)
        latex_1_matrix = sp.latex(sp.Matrix(K_split[0]))
        latex_2_matrix = sp.latex(sp.Matrix(K_split[1]))
        latex_3_matrix = sp.latex(sp.Matrix(K_split[2]))
        f.write("Stiffness Matrix (Part 1):\n")
        f.write(latex_1_matrix + "\n\n")
        f.write("Stiffness Matrix (Part 2):\n")
        f.write(latex_2_matrix + "\n\n")
        f.write("Stiffness Matrix (Part 3):\n")
        f.write(latex_3_matrix + "\n\n")

        latex_matrix = sp.latex(sp.Matrix(M_evals))
        f.write("Mass Matrix Evals:\n")
        f.write(latex_matrix + "\n\n")
        
        latex_matrix = sp.latex(sp.Matrix(K_evals))
        f.write("Stiffness Matrix Evals:\n")
        f.write(latex_matrix + "\n\n")

        latex_matrix = sp.latex(sp.Matrix(evals))
        f.write("M^-1 * K Evals:\n")
        f.write(latex_matrix + "\n\n")
        
        f.write(f"Maximum Eigenvalue of M^-1 * K: {max_eigenvalue}\n")
    
