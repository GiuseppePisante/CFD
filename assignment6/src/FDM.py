import numpy as np
import matplotlib.pyplot as plt
from solver import solver_FDM

def analytical_solution(x, Pe):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

def compute_error_norms(phi_num, phi_exact):
    L1 = np.sum(np.abs(phi_num - phi_exact)) / len(phi_num)
    L2 = np.sqrt(np.sum((phi_num - phi_exact) ** 2) / len(phi_num))
    return L1, L2

def solve_for_N(N, Pe):
    n = N - 1  
    x = np.linspace(0, 1, N + 1) ** 0.7  
    
    A, b = solver_FDM(n, x, Pe)
    phi_num_internal = np.linalg.solve(A, b)
    
    phi_exact_internal = analytical_solution(x[1:-1], Pe)  

    phi_bc_left = analytical_solution(np.array([x[0]]), Pe)[0]  
    phi_bc_right = analytical_solution(np.array([x[-1]]), Pe)[0] 

    phi_num = np.concatenate(([phi_bc_left], phi_num_internal, [phi_bc_right]))
    phi_exact = np.concatenate(([phi_bc_left], phi_exact_internal, [phi_bc_right]))

    L1, L2 = compute_error_norms(phi_num_internal, phi_exact_internal)

    return x, phi_num, phi_exact, L1, L2

# Parameters
Pe = 25  
N_values = [31, 63, 127]
errors_L1 = []
errors_L2 = []

plt.figure(figsize=(8, 5))
for N in N_values:
    x, phi_num, phi_exact, L1, L2 = solve_for_N(N, Pe)
    errors_L1.append(L1)
    errors_L2.append(L2)

    plt.plot(x, phi_num, 'o-', label=f'N={N}')
    plt.plot(x, phi_exact, '--', label='Analytical')
    plt.xlabel('x')
    plt.ylabel('phi(x)')
    plt.title('Numerical Solution for Different N')
    plt.legend()
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