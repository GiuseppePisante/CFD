import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver
from matplotlib.animation import FuncAnimation

def heat_transfer_pde(T, t, x, y):
    """
    T: temperature
    t: time
    x, y: spatial coordinates
    alpha: thermal diffusivity
    """
    # Assuming 2D heat equation: dT/dt = (d^2T/dx^2 + d^2T/dy^2)
    dTdx2 = solver.second_derivative(T, x[0, :], axis=1)
    dTdy2 = solver.second_derivative(T, y[:, 0], axis=0)
    dTdt = dTdx2 + dTdy2
    return dTdt

# Create the mesh
mesh_instance = mesh.Mesh(x_start=0, x_end=1, y_start=0, y_end=1, x_points=100, y_points=100)
mesh_instance.mesh_generator()
x, y = mesh_instance.x, mesh_instance.y

# Initial condition
T_initial = np.zeros_like(x)  # T(x, y, 0) = 0

# Boundary conditions
T_initial[0, :] = 1 - y[0, :]**3  # T(0, y, 0) = 1 - y^3
T_initial[-1, :] = 1 - np.sin(np.pi * y[0, :] / 2)  # T(1, y, t) = 1 - sin(pi*y/2)
T_initial[:, 0] = 1  # T(x, 0, t) = 1
T_initial[:, -1] = 0  # T(x, 1, t) = 0

# Time parameters
t_start = 0
t_end = 0.16
num_time_steps = 100  # Increase the number of time steps for better stability

# Solve the PDE and store all time steps using Crank-Nicolson method
dt = (t_end - t_start) / num_time_steps
T = T_initial.copy()
t = t_start
T_all_steps = [T.copy()]

for _ in range(num_time_steps):
    T = solver.Crank_Nicolson(T, x, y, t, t + dt, 1)
    t += dt
    T_all_steps.append(T.copy())

# Create the animation
fig, ax = plt.subplots()
contour = ax.contourf(x, y, T_all_steps[0], cmap='hot')
plt.colorbar(contour, ax=ax, label='Temperature')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Heat Transfer PDE Solution')

def update(frame):
    ax.clear()
    contour = ax.contourf(x, y, T_all_steps[frame], cmap='hot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Heat Transfer PDE Solution at t={frame*dt:.4f}')
    return contour,

ani = FuncAnimation(fig, update, frames=num_time_steps, blit=False)
plt.show()