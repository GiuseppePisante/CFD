import numpy as np
import matplotlib.pyplot as plt
from solver import solve_CDS

# Parameters
L = 1.0            
Pe = 25            
U = 1.0             
rho = 1.0           
alpha = U * L / Pe  

# Grid spacing
Nx = 32             
Dx = L / Nx       
x = np.linspace(0, L, Nx)

# Time step
d = alpha / (rho * Dx**2)  
c = U / Dx                 
Dt = min(0.5 / d, 1.0 / c) 
nt_max = 10000            

# Initial and Boundary condition 
phi = np.zeros(Nx)        
phi[-1] = 1
phi_new = np.zeros_like(phi)

# Time integration with CDS scheme
phi, time, steps, steady_state = solve_CDS(phi, Dt, Dx, Pe, nt_max)

if steady_state:
    print(f"Steady state reached at time {time:.2f} after {steps} time steps.")
else:
    print("Maximum number of time steps reached before steady state.")

# Plot the result
plt.plot(x, phi, label="Numerical Solution")
plt.xlabel("x")
plt.ylabel("Phi")
plt.title("Steady-state Advection-Diffusion")
plt.legend()
plt.grid()
plt.show()