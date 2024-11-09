import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize



def heat_transfer_pde(T, t, x, y):
    # 2D heat equation: dT/dt = (d^2T/dx^2 + d^2T/dy^2)
    dTdx2 = solver.second_derivative(T, x[0, :], axis=1)
    dTdy2 = solver.second_derivative(T, y[:, 0], axis=0)
    dTdt = dTdx2 + dTdy2
    return dTdt


# Create the mesh
mesh_instance = mesh.Mesh(x_start=0, x_end=1, y_start=0, y_end=1, x_points=40, y_points=40)
mesh_instance.mesh_generator()
x, y = mesh_instance.x, mesh_instance.y

# Initial condition
T_initial = np.zeros_like(x)

# Time parameters
t_start = 0
t_end = 0.16
num_time_steps = 1600 # 16 160 1600 # ! con 16 è davvero orribile

# Solve the PDE and store all time steps using Explicit-Euler method
dt = (t_end - t_start) / num_time_steps
T_explicit = T_initial.copy()
T_crank = T_initial.copy()
t = t_start
T_explicit_all_steps = [T_explicit.copy().T]
T_crank_all_steps = [T_crank.copy().T]

for _ in range(num_time_steps):
    T_explicit = solver.explicit_euler(heat_transfer_pde, T_explicit, x, y, t, dt)
    T_crank = solver.crank_nicolson(heat_transfer_pde, T_crank, x, y, t, dt)
    t += dt
    T_explicit_all_steps.append(T_explicit.copy().T)
    T_crank_all_steps.append(T_crank.copy().T)

# Compute the global min and max temperature for consistent color mapping
vmin = min(np.min(T_explicit_all_steps), np.min(T_crank_all_steps))
vmax = max(np.max(T_explicit_all_steps), np.max(T_crank_all_steps))

# Define normalization and color levels
norm = Normalize(vmin=vmin, vmax=vmax)
levels = np.linspace(vmin, vmax, 100)

# Create the initial plot with consistent color scaling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
contour1 = ax1.contourf(x, y, T_explicit_all_steps[0], cmap='hot', levels=levels, origin='lower', norm=norm)
contour2 = ax2.contourf(x, y, T_crank_all_steps[0], cmap='hot', levels=levels, origin='lower', norm=norm)
plt.colorbar(contour1, ax=ax1, label='Temperature')
plt.colorbar(contour2, ax=ax2, label='Temperature')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Explicit Euler Solution')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Crank-Nicolson Solution')

def update(frame):
    # Clear previous contours only, not the entire axis
    for c in ax1.collections:
        c.remove()
    for c in ax2.collections:
        c.remove()
    
    # Update contours with the current frame and keep color scaling consistent
    ax1.contourf(x, y, T_explicit_all_steps[frame], cmap='hot', levels=levels, origin='lower', norm=norm)
    ax2.contourf(x, y, T_crank_all_steps[frame], cmap='hot', levels=levels, origin='lower', norm=norm)
    
    # Update titles with time information
    ax1.set_title(f'Explicit Euler Solution at t={frame*dt:.4f}')
    ax2.set_title(f'Crank-Nicolson Solution at t={frame*dt:.4f}')

ani = FuncAnimation(fig, update, frames=num_time_steps, blit=False)
ani.save('heat_transfer_animation.mp4', writer='ffmpeg', fps=30)