import numpy as np
import matplotlib.pyplot as plt
import solver as solv

# Parameters
L = 1.0
U = 1.0
rho = 1.0
Pe_values = [25, np.inf]  # Use Pe = 25 and Pe -> infinity
dx_values = [1/16, 1/32, 1/64]
k_values = [1, 2, 4, 8, 16]  # Different wave numbers

# Time parameters
dt = 0.01
nt_max = 10000

# Periodic Boundary Solver
def apply_periodic_boundary(phi):
    phi[0] = phi[-1]
    return phi

# Run simulations
fig, axs = plt.subplots(len(k_values), len(dx_values), figsize=(20, 15), sharex=True, sharey=True)

for i, k in enumerate(k_values):
    for j, dx in enumerate(dx_values):
        N = int(L / dx) + 1
        x = np.linspace(0, L, N)
        phi_initial = np.sin(2 * k * np.pi * x)  # Initial condition
        
        for Pe in Pe_values:
            # Initialize phi
            phi = phi_initial.copy()
            
            # Time-stepping loop for periodic conditions
            for _ in range(nt_max):
                phi = solv.CDS(phi, dt, dx, Pe, nt_max)  # Update phi using solver
                phi = apply_periodic_boundary(phi)  # Apply periodic boundary condition

            # Plot results
            axs[i, j].plot(x, phi, label=f'Pe={Pe}')
        
        axs[i, j].set_title(f'k={k}, dx={dx}')
        axs[i, j].legend()
        axs[i, j].set_xlabel('x')
        axs[i, j].set_ylabel('Phi')

plt.tight_layout()
plt.savefig('task4_periodic_fixed.png')
plt.show()
