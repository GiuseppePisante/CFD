import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver
from matplotlib.colors import Normalize
from scipy.sparse.linalg import bicgstab, LinearOperator, spilu

def main():
    L = 1.0 
    Nx, Ny = 20, 20
    Re = 10000.0 

    delta_99=4.91/np.sqrt(Re)*L
    H = 2.0 * delta_99
    dx, dy = H/Nx, L/Ny 

    # Create an instance of LinearSystem
    linear_system = solver.LinearSystem(Nx, Ny, dx, dy, Re)
    
    # Build the initial matrix
    linear_system.build_matrix()

    # Visualize the sparsity pattern of the final matrix A
    plt.spy(linear_system.A, markersize=5)
    plt.title("Sparsity pattern of matrix A")
    plt.show()
        
    # Initial guess for u and v
    solution = np.ones((2*Nx* Ny))

    tolerance = 1.e-10

    for iteration in range(10):
        print(f"Iteration {iteration + 1}")
        
        # Solve the linear system with BiCGSTAB and block preconditioner
        solution, info = bicgstab(linear_system.A, linear_system.b)

        # Update the matrix A and vector b using the current solution
        linear_system.updateMatrix(solution)

        # Check convergence
        residual = np.linalg.norm(linear_system.b - linear_system.A @ solution)
        normalized_residual = residual / np.linalg.norm(linear_system.b)
        print("normalized residual:", normalized_residual)
        if normalized_residual < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
        
    # Estrazione delle soluzioni u e v
    u_solution = solution[:Nx * Ny]
    v_solution = solution[Nx * Ny:]

    u_solution = u_solution.reshape((Ny, Nx))
    v_solution = v_solution.reshape((Ny, Nx))

    #PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    u_min, u_max = np.min(u_solution), np.max(u_solution)
    levels = np.linspace(u_min, u_max, 100)
    
    norm = Normalize(vmin=0, vmax=1)
    
    contour1 = ax1.contourf(np.arange(Ny)*dy, np.arange(Nx)*dx, u_solution, cmap='coolwarm', levels=levels, origin='lower', norm=norm)
    contour2 = ax2.contourf(np.arange(Ny)*dy, np.arange(Nx)*dx, v_solution, cmap='coolwarm', levels=levels, origin='lower', norm=norm)
    
    plt.colorbar(contour1, ax=ax1, label='u velocity')
    plt.colorbar(contour2, ax=ax2, label='v velocity')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('u velocity field')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('v velocity field')
    
    plt.tight_layout()
    plt.show()
    
    """ print("Solution:", solution)
    print("GMRES info:", info) """

if __name__ == "__main__":
    main()



