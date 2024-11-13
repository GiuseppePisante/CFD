import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver

def heat_transfer_pde(T, t, x, y):
    # 2D heat equation: dT/dt = (d^2T/dx^2 + d^2T/dy^2)
    dTdx2 = solver.second_derivative(T, x[0, :], axis=1)
    dTdy2 = solver.second_derivative(T, y[:, 0], axis=0)
    dTdt = dTdx2 + dTdy2
    return dTdt

# Create the mesh using dx = dy = 1/40 and 1/100
mesh_instance = mesh.Mesh(x_start=0, x_end=1, y_start=0, y_end=1, x_points=100, y_points=100)
mesh_instance.mesh_generator()
x, y = mesh_instance.x, mesh_instance.y

# Initial condition
T_initial = np.zeros_like(x)

# Time parameters
t_start = 0
t_end = 0.16
time_steps = 6400

# Solve the PDE and store all time steps using Explicit-Euler method
dt = (t_end - t_start) / time_steps
T_explicit = T_initial.copy()
T_plot = []
t = t_start

for _ in range(time_steps):
    T_explicit = solver.explicit_euler(heat_transfer_pde, T_explicit, x, y, t, dt)
    t += dt

    if _ in [399, 799, 1599, 3199, 6399]:
        # [9, 19, 39, 79, 159] for time_steps = 160
        # [99, 199, 399, 799, 1599] for time_steps = 1600
        # [399, 799, 1599, 3199, 6399] for time_steps = 6400 
        T_plot.append(T_explicit)
        
# Plotting the numerical solution of temperature over the whole domain at times t = 0.01, 0.02, 0.04, 0.08, 0.16. 
times = [0.01, 0.02, 0.04, 0.08, 0.16]  
fig, axs = plt.subplots(1, 5, figsize=(15, 4))

for i, ax in enumerate(axs):
    contour = ax.contourf(x[0, :], y[:, 0], T_plot[i].T, cmap='hot')
    ax.set_title(f"Time = {times[i]}s")
    fig.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()