import numpy as np
import matplotlib.pyplot as plt

# Parameters  
L = 1.0
Re = 10000                   
delta_99 = 4.91 / np.sqrt(Re) * L  
H = 2.0 * delta_99  

# Grid
Nx = 500      
Ny = 250     
dx = L / Nx   
dy = H / Ny      

# Stability condition
dt = min(dx**2, dy**2) / (2 * Re)

x = np.linspace(0, L, Nx)
y = np.linspace(0, H, Ny)
u = np.zeros((Nx, Ny))  # Velocità u
v = np.zeros((Nx, Ny))  # Velocità v

# Boundary conditions
u[:, -1] = 1  # Free-stream condition
u[:, 0] = 0   # No-slip condition
v[:, 0] = 0   # No-penetration condition

# Solution
max_iter = 500

for n in range(max_iter):
    u_old = u.copy()
    v_old = v.copy()
    
    # Momentum equation
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            conv_x = -u_old[i, j] * (u_old[i+1, j] - u_old[i-1, j]) / (2 * dx)
            conv_y = -v_old[i, j] * (u_old[i, j+1] - u_old[i, j-1]) / (2 * dy)
            diff_y = (u_old[i, j+1] - 2 * u_old[i, j] + u_old[i, j-1]) / (dy**2)
            
            u[i, j] = u_old[i, j] + dt * (conv_x + conv_y + (1 / Re) * diff_y)
    
    # Continuity equation
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            v[i, j] = v_old[i, j-1] - dy * (u_old[i+1, j] - u_old[i-1, j]) / (2 * dx)
    
    # Boundary conditions
    u[:, -1] = 1  # Free-stream condition
    u[:, 0] = 0   # No-slip condition
    v[:, 0] = 0   # No-penetration condition

# Plot
X, Y = np.meshgrid(x, y)
plt.figure(figsize=(10, 6))
cp = plt.contourf(X.T, Y.T, u, levels=50, cmap="jet")
plt.colorbar(cp)
plt.title("Distribuzione della velocità u(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()