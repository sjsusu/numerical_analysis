import numpy as np
import sympy as sp

def global_indexing(width, height=None, include_boundary=False):
    if height is None:
        height = width
    if include_boundary:
        return np.arange(width * height).reshape((width, height))
    return np.arange((width-2)*(height-2)).reshape((width-2, height-2))

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

def det_jacobian(xe, ye):
    return (1/4) * abs(xe[1] - xe[0]) * abs(ye[2] - ye[1])

def element_mass_matrix(xe, ye):
    detJ = det_jacobian(xe, ye)
    Me = (detJ / 9) * np.array([[4, 2, 1, 2],
                                 [2, 4, 2, 1],
                                 [1, 2, 4, 2],
                                 [2, 1, 2, 4]])
    return Me

def element_stiffness_matrix(xe, ye):
    detJ = det_jacobian(xe, ye)
    Ke = detJ *(4 / 6) * np.array([[4, -1, -2, -1],
                                 [-1, 4, -1, -2],
                                 [-2, -1, 4, -1],
                                 [-1, -2, -1, 4]])
    return Ke

def generate_global_coordinates(width, height=None):
    if height is None:
        height = width
    
    # Generate global coordinates for interior nodes
    # given Dirichlet BCs
    x_nodes = np.linspace(-2, 2, width)
    x_nodes = x_nodes[1:-1]
    y_nodes = np.linspace(-2, 2, height)
    y_nodes = y_nodes[1:-1]

    # Create meshgrid of coordinates
    x_mesh, y_mesh = np.array(np.meshgrid(x_nodes, y_nodes))
    # Reshape as list of (x, y) pairs
    global_coordinates = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    return global_coordinates


def global_assembly(width, height=None):
    if height is None:
        height = width
        
    global_coordinates = generate_global_coordinates(width, height)
    
    global_indices = global_indexing(width, height)
    connectivity_matrix = generate_connectivity_matrix(global_indices)

    num_nodes = (width - 2) * (height - 2)
    M_global = np.zeros((num_nodes, num_nodes))
    K_global = np.zeros((num_nodes, num_nodes))
    
    for element in range(connectivity_matrix.shape[0]):
        # Get the global node indices for this element
        element_coordinates = connectivity_matrix[element, :]
        
        # Extract x and y coordinates for element
        # x coordinates
        xe = global_coordinates[element_coordinates, 0]  
        # y coordinates
        ye = global_coordinates[element_coordinates, 1] 
        
        # Compute element matrices
        Me = element_mass_matrix(xe, ye)
        Ke = element_stiffness_matrix(xe, ye)
        
        # Add element contributions to global matrices
        for i_local in range(4):
            i_global = element_coordinates[i_local]
            for j_local in range(4):
                j_global = element_coordinates[j_local]
                M_global[i_global, j_global] += Me[i_local, j_local]
                K_global[i_global, j_global] += Ke[i_local, j_local]
    
    return global_coordinates, M_global, K_global

if __name__ == "__main__":
    width = 5  # Number of nodes along one dimension
    global_indices = global_indexing(width)
    connectivity_matrix = generate_connectivity_matrix(global_indices)
    _, M, K = global_assembly(width)
    
    global_with_boundary = global_indexing(width, include_boundary=True)
    connectivity_matrix_with_boundary = generate_connectivity_matrix(global_with_boundary)
    
    M_inverse = np.linalg.inv(M)
    M_inv_K = np.dot(M_inverse, K)
    evals = np.linalg.eigvals(M_inv_K)
    max_eigenvalue = np.max(np.abs(evals))

    with open("./outputs_4/matrices.txt", "w") as f:
        latex_matrix = sp.latex(sp.Matrix(connectivity_matrix_with_boundary))
        f.write("Connectivity Matrix with Boundary:\n")
        f.write(latex_matrix + "\n\n")
        
        print(global_indices)
        latex_matrix = sp.latex(sp.Matrix(global_indices))
        f.write("Global Indices Matrix:\n")
        f.write(latex_matrix + "\n\n")
        
        print(connectivity_matrix)
        latex_matrix = sp.latex(sp.Matrix(connectivity_matrix))
        f.write("Connectivity Matrix:\n")
        f.write(latex_matrix + "\n\n")

        M = np.round(M, 4)
        latex_matrix = sp.latex(sp.Matrix(M))
        f.write("Mass Matrix:\n")
        f.write(latex_matrix + "\n\n")

        K = np.round(K, 4)
        latex_matrix = sp.latex(sp.Matrix(K))
        f.write("Stiffness Matrix:\n")
        f.write(latex_matrix + "\n\n")
        
        latex_matrix = sp.latex(sp.Matrix(evals))
        f.write("M^-1 * K Evals:\n")
        f.write(latex_matrix + "\n\n")
        
        f.write(f"Maximum Eigenvalue of M^-1 * K: {max_eigenvalue}\n")
