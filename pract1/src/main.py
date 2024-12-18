import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import time  # For timing execution

def heat_transfer_pde(T, t, x, y):
    # 2D heat equation: dT/dt = (d^2T/dx^2 + d^2T/dy^2)
    dTdx2 = solver.second_derivative(T, x[0, :], axis=1)
    dTdy2 = solver.second_derivative(T, y[:, 0], axis=0)
    dTdt = dTdx2 + dTdy2
    return dTdt

# Create the mesh
mesh_instance = mesh.Mesh(x_start=0, x_end=1, y_start=0, y_end=1, x_points=100, y_points=40)
mesh_instance.mesh_generator()
x, y = mesh_instance.x, mesh_instance.y

# Initial condition
T_initial = np.zeros_like(x)

# Time parameters
t_start = 0
t_end = 0.16
num_time_steps = 6400

# Solve the PDE and store all time steps using Explicit-Euler and Crank-Nicolson methods
dt = (t_end - t_start) / num_time_steps
T_explicit = T_initial.copy()
T_crank = T_initial.copy()
T_explicit_all_steps = [T_explicit.copy().T]
T_crank_all_steps = [T_crank.copy().T]

# Factorize A matrix for Crank-Nicolson
P, L, U = solver.FactorizeA(T_initial, dt, x[0, 1] - x[0, 0], y[1, 0] - y[0, 0])

# Timing Explicit-Euler method
start_explicit = time.time()
t = t_start
for _ in range(num_time_steps):
    T_explicit = solver.explicit_euler(heat_transfer_pde, T_explicit, x, y, t, dt)
    t += dt
    T_explicit_all_steps.append(T_explicit.copy().T)
end_explicit = time.time()
time_explicit = end_explicit - start_explicit
print(f"Explicit Euler method took {time_explicit:.4f} seconds for {num_time_steps} steps.")

# Timing Crank-Nicolson method
start_crank = time.time()
t = t_start
for _ in range(num_time_steps):
    T_crank = solver.crank_nicolson(heat_transfer_pde, T_crank, x, y, t, dt, P, L, U)
    t += dt
    T_crank_all_steps.append(T_crank.copy().T)
end_crank = time.time()
time_crank = end_crank - start_crank
print(f"Crank-Nicolson method took {time_crank:.4f} seconds for {num_time_steps} steps.")


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
    ax1.contourf(x, y, T_explicit_all_steps[frame], cmap='hot', origin='lower', norm=norm)
    ax2.contourf(x, y, T_crank_all_steps[frame], cmap='hot', origin='lower', norm=norm)
    
    # Update titles with time information
    ax1.set_title(f'Explicit Euler Solution at t={frame*dt:.4f}')
    ax2.set_title(f'Crank-Nicolson Solution at t={frame*dt:.4f}')

ani = FuncAnimation(fig, update, frames=num_time_steps, blit=False)
plt.show()
#ani.save('heat_transfer_animation.mp4', writer='ffmpeg', fps=30)