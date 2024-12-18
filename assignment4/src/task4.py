import numpy as np
import matplotlib.pyplot as plt
import solver4 as solv

# Parameters
L = 1.0                       
U = 1.0             
rho = 1.0
Pe_values = np.array([25, 50, 100, 120, 125, 128, 130, 135, 150, 200, 400])

# Grid spacings
dx = 1/32
N = int(1/dx) + 1
x = np.linspace(0, L, N)   
dt = 0.001    
nt_max= 1000         

# Initial and boundary condition 
phi = np.zeros(N)        
phi[-1] = 1

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, Pe in enumerate([25, 50, 100]):
  
    phi_1 = solv.CDS(phi, dt, dx, Pe, nt_max)
    axs[0, 0].plot(x, phi_1, label=f'Pe={Pe}')
    axs[0, 0].set_title('discretization1')
    axs[0, 0].legend()

    phi_2 = solv.UP1(phi, dt, dx, Pe, nt_max)
    axs[1, 0].plot(x, phi_2, label=f'Pe={Pe}')
    axs[1, 0].set_title('discretization2')
    axs[1, 0].legend()
    
    phi_3 = solv.UP2(phi, dt, dx, Pe, nt_max)
    axs[2, 0].plot(x, phi_3, label=f'Pe={Pe}')
    axs[2, 0].set_title('discretization3')
    axs[2, 0].legend()

for i, Pe in enumerate([120, 125, 128, 130, 135]):
  
    phi_1 = solv.CDS(phi, dt, dx, Pe, nt_max)
    axs[0, 1].plot(x, phi_1, label=f'Pe={Pe}')
    axs[0, 1].set_title('discretization1')
    axs[0, 1].legend()

    phi_2 = solv.UP1(phi, dt, dx, Pe, nt_max)
    axs[1, 1].plot(x, phi_2, label=f'Pe={Pe}')
    axs[1, 1].set_title('discretization2')
    axs[1, 1].legend()

    phi_3 = solv.UP2(phi, dt, dx, Pe, nt_max)
    axs[2, 1].plot(x, phi_3, label=f'Pe={Pe}')
    axs[2, 1].set_title('discretization3')
    axs[2, 1].legend()

for i, Pe in enumerate([150, 200, 400]):
  
    phi_1 = solv.CDS(phi, dt, dx, Pe, nt_max)
    axs[0, 2].plot(x, phi_1, label=f'Pe={Pe}')
    axs[0, 2].set_title('discretization1')
    axs[0, 2].legend()

    phi_2 = solv.UP1(phi, dt, dx, Pe, nt_max)
    axs[1, 2].plot(x, phi_2, label=f'Pe={Pe}')
    axs[1, 2].set_title('discretization2')
    axs[1, 2].legend()

    phi_3 = solv.UP2(phi, dt, dx, Pe, nt_max)
    axs[2, 2].plot(x, phi_3, label=f'Pe={Pe}')
    axs[2, 2].set_title('discretization3')
    axs[2, 2].legend()

plt.tight_layout()
plt.savefig('task4.png')