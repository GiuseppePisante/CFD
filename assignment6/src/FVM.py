import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def analytical_solution(x, Pe):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

def compute_error_norms(phi_num, phi_exact):
    L1 = np.sum(np.abs(phi_num - phi_exact)) / len(phi_num)
    L2 = np.sqrt(np.sum((phi_num - phi_exact) ** 2) / len(phi_num))
    return L1, L2

def solve_fvm(N, Pe, u):
    # Generate non-uniform grid
    x = np.linspace(0, 1, N) ** 0.7
    dx = np.diff(x)  # Control volume sizes
    
    # Compute face-centered grid points
    xf = (x[:-1] + x[1:]) / 2
    
    # Compute interpolation weights
    w = dx[:-1] / (x[1:-1] - xf[:-1])
    w_prime = dx[1:] / (x[2:] - xf[1:])
    
    # Coefficients for tridiagonal system
    a_lower = -1 / (Pe * dx[:-1]) - u * w_prime  # Sub-diagonal
    a_main = (1 / (Pe * dx[:-1]) + 1 / (Pe * dx[1:])) + u * w - u * (1 - w_prime)  # Main diagonal
    a_upper = -1 / (Pe * dx[1:]) - u * (1 - w)  # Super-diagonal
    
    # Set up banded matrix for solving Ax = b
    A_banded = np.zeros((3, N - 2))
    A_banded[0, 1:] = a_upper[:-1]  # Super-diagonal (shifted up)
    A_banded[1, :] = a_main  # Main diagonal
    A_banded[2, :-1] = a_lower[1:]  # Sub-diagonal (shifted down)
    
    # Right-hand side vector
    b = np.zeros(N - 2)
    
    # Apply Dirichlet boundary conditions
    Phi_0, Phi_N = 0.0, 1.0  # Boundary values
    b[0] -= a_lower[0] * Phi_0
    b[-1] -= a_upper[-1] * Phi_N
    
    # Solve the system
    Phi_inner = solve_banded((1, 1), A_banded, b)
    
    # Full solution including boundary conditions
    Phi_num = np.hstack([Phi_0, Phi_inner, Phi_N])
    return x, Phi_num

N_values = [32, 64, 128]  # Number of points including boundaries (faces = N-1)
Pe = 25  # Peclet number
u = 0.1
errors_L1 = []
errors_L2 = []

for N in N_values:
    x, Phi_num = solve_fvm(N, Pe, u)
    Phi_exact = analytical_solution(x, Pe)
    L1, L2 = compute_error_norms(Phi_num, Phi_exact)
    errors_L1.append(L1)
    errors_L2.append(L2)
    
    # Plot numerical vs analytical solution
    plt.plot(x, Phi_num, 'o-', label=f'Numerical (N={N})')
    plt.plot(x, Phi_exact, '--', label='Analytical')
    plt.xlabel('x')
    plt.ylabel('Φ')
    plt.title(f'Finite-Volume Solution (N={N}, Pe={Pe})')
    plt.legend()
    plt.grid()
    plt.show()

# Compute order of accuracy
orders_L1 = [np.log(errors_L1[i-1] / errors_L1[i]) / np.log(2) for i in range(1, len(errors_L1))]
orders_L2 = [np.log(errors_L2[i-1] / errors_L2[i]) / np.log(2) for i in range(1, len(errors_L2))]

print("L1 Errors:", errors_L1)
print("L2 Errors:", errors_L2)
print("Estimated Order of Accuracy (L1):", orders_L1)
print("Estimated Order of Accuracy (L2):", orders_L2)

# Plot L1 and L2 errors
plt.figure(figsize=(8, 5))
plt.loglog(N_values, errors_L1, 'o-', label='L1 Error')
plt.loglog(N_values, errors_L2, 's-', label='L2 Error')
plt.xlabel('N')
plt.ylabel('Error Norm')
plt.title('L1 and L2 Error Norms')
plt.legend()
plt.show()
