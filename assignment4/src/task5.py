import numpy as np
import matplotlib.pyplot as plt
import solver5 as solv  

# Grid spacings
dx_values = [1/16, 1/32, 1/64]
Pe_values = [25, 1e6]  
k_values = [1, 2, 4, 8, 16]
dt = 0.001
nt_max = 1000

# Periodic boundary condition 
def periodic_bc(phi):
    phi[0] = phi[-2]  
    phi[-1] = phi[1] 
    return phi

for dx in dx_values:
    N = int(1 / dx) + 1
    x = np.linspace(0, 1, N)

    fig, axs = plt.subplots(1, len(k_values), figsize=(20, 5), sharey=True)
    fig.suptitle(f"Solutions for dx = {dx:.3f}", fontsize=16)

    for j, k in enumerate(k_values):
        phi = np.sin(2 * k * np.pi * x)
        phi = periodic_bc(phi)

        for Pe in Pe_values:
            phi_CDS = solv.CDS(phi.copy(), dt, dx, Pe, nt_max)
            phi_UP1 = solv.UP1(phi.copy(), dt, dx, Pe, nt_max)
            phi_UP2 = solv.UP2(phi.copy(), dt, dx, Pe, nt_max)

            label = f"Pe={Pe:.0f}"
            axs[j].plot(x, phi_CDS, label=f"CDS, {label}")
            axs[j].plot(x, phi_UP1, label=f"UP1, {label}")
            axs[j].plot(x, phi_UP2, label=f"UP2, {label}")
        
        axs[j].set_title(f"k={k}")
        axs[j].set_xlabel("x")
        axs[j].legend()
        axs[j].grid()

    axs[0].set_ylabel("Φ(x)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 

    plt.savefig(f"task5_dx_{int(1/dx)}.png")