import numpy as np
import matplotlib.pyplot as plt
from solver import solver_FVM

def analytical_solution(x, Pe):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

def compute_error_norms(phi_num, phi_exact):
    L1 = np.sum(np.abs(phi_num - phi_exact)) / len(phi_num)
    L2 = np.sqrt(np.sum((phi_num - phi_exact) ** 2) / len(phi_num))
    return L1, L2

# Parameters
N_values = [32, 64, 128]  
Pe = 25  
u = 0.1
errors_L1 = []
errors_L2 = []

for N in N_values:
    x, Phi_num = solver_FVM(N, Pe, u)
    Phi_exact = analytical_solution(x, Pe)
    L1, L2 = compute_error_norms(Phi_num, Phi_exact)
    errors_L1.append(L1)
    errors_L2.append(L2)
    
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