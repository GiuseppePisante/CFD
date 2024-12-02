import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from scipy.sparse.linalg import gmres

def main():
    Nx, Ny = 10, 10  # Example grid size
    dx, dy = 1.0, 1.0  # Example grid spacing
    Re = 100.0  # Example Reynolds number

    # Create an instance of LinearSystem
    linear_system = solver.LinearSystem(Nx, Ny, dx, dy, Re)
    
    # Build the initial matrix
    linear_system.build_matrix()

    # Visualize the sparsity pattern of the final matrix A
    plt.spy(linear_system.A, markersize=5)
    plt.title("Sparsity pattern of matrix A")
    plt.show()
        
    # Initial guess for u and v
    u_initial = np.ones((Nx, Ny))
    v_initial = np.ones((Nx, Ny))

    # Set initial conditions on the boundaries
    u_initial[:, 0] = 1.0  # Left boundary
    u_initial[:, -1] = 1.0  # Right boundary
    u_initial[0, :] = 0.0  # Bottom boundary
    u_initial[-1, :] = 1.0  # Top boundary

    v_initial[0, :] = 0.0  # Bottom boundary
    v_initial[-1, :] = 0.0  # Top boundary

    # Define a callback function to update the matrix at each iteration
    def gmres_callback(xk):
       linear_system.updateMatrix(u_initial, v_initial)

    
    # Solve the system using GMRES
    solution, info = gmres(linear_system.A, linear_system.b, callback=gmres_callback, callback_type='legacy')
    
    # Extract u and v from the solution
    u_solution = solution[:Nx * Ny].reshape((Nx, Ny))
    v_solution = solution[Nx * Ny:].reshape((Nx, Ny))
    
    # Create the initial plot with consistent color scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    levels = np.linspace(np.min(u_solution), np.max(u_solution), 100)
    norm = Normalize(vmin=np.min(u_solution), vmax=np.max(u_solution))
    
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



