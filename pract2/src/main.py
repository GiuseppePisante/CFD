import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver
from matplotlib.colors import Normalize
from scipy.sparse.linalg import gmres, LinearOperator, spilu

def main():
    L = 1.0 
    Nx, Ny = 10, 10
    dx, dy = L/Nx, L/Ny 
    Re = 1000.0 

    # Create an instance of LinearSystem
    linear_system = solver.LinearSystem(Nx, Ny, dx, dy, Re)
    
    # Build the initial matrix
    linear_system.build_matrix()

    # Visualize the sparsity pattern of the final matrix A
    plt.spy(linear_system.A, markersize=5)
    plt.title("Sparsity pattern of matrix A")
    plt.show()
        
    # Initial guess for u and v
    u_solution = np.zeros((Nx, Ny))
    v_solution = np.zeros((Nx, Ny))
    u_new = np.zeros((Nx, Ny))
    v_new = np.zeros((Nx, Ny))

    tolerance = 1.e-10

    for iteration in range(10):
        print(f"Iteration {iteration + 1}")
        
        # Solve the linear system with GMRES
        solution, info = gmres(linear_system.A, linear_system.b)

        # Extract u and v from the solution
        u_solution = solution[:Nx * Ny].reshape((Nx, Ny))
        v_solution = solution[Nx * Ny:].reshape((Nx, Ny))

        # Update the matrix A and vector b using the current solution
        linear_system.updateMatrix(u_solution, v_solution)

        # Check convergence
        residual = np.linalg.norm(linear_system.b - linear_system.A @ solution)
        normalized_residual = residual / np.linalg.norm(linear_system.b)
        print("normalized residual:", normalized_residual)
        if normalized_residual < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

    


    #PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    u_min, u_max = np.min(u_solution), np.max(u_solution)
    levels = np.linspace(u_min, u_max, 100)
    
    norm = Normalize(vmin=u_min, vmax=u_max)
    
    contour1 = ax1.contourf(np.arange(Nx)*dx, np.arange(Ny)*dy, u_solution, cmap='coolwarm', levels=levels, origin='lower', norm=norm)
    contour2 = ax2.contourf(np.arange(Nx)*dx, np.arange(Ny)*dy, v_solution, cmap='coolwarm', levels=levels, origin='lower', norm=norm)
    
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



