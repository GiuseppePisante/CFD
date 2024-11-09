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

# Point 1 of tas 1.3
T_explicit_point = [T_explicit[9,9]]
T_crank_point = [T_crank[9,9]]

# Point 2 
T_explicit_final = []
T_crank_final = []


for _ in range(num_time_steps):
    T_explicit = solver.explicit_euler(heat_transfer_pde, T_explicit, x, y, t, dt)
    T_crank = solver.crank_nicolson(heat_transfer_pde, T_crank, x, y, t, dt)
    t += dt
    T_explicit_point.append(T_explicit[9,9])
    T_crank_point.append(T_crank[9,9])

T_explicit_final = np.array(T_explicit[9, :])
T_crank_final = np.array(T_crank[9, :])


# Create a time array based on dt and num_time_steps
time_points = np.linspace(t_start, t_end, num_time_steps + 1)  # +1 to include t=0

# Plot temperature evolution at (9,9) for both methods
plt.figure(figsize=(10, 6))
plt.plot(time_points, T_explicit_point, label='Explicit Euler', marker='o')
plt.plot(time_points, T_crank_point, label='Crank-Nicolson', marker='x')
plt.xlabel('Time (t)')
plt.ylabel('Temperature at (x, y) = (0.4, 0.4)')
plt.title('Temperature Evolution at (x, y) = (0.4, 0.4)')
plt.legend()
plt.grid(True)
plt.savefig("temperature_evolution_point_0404.png")


y_points = np.linspace(0, 1, 40)  # 40 points based on delta_y = 1/40

# Plotting the vertical temperature profile along y at x = 0.4
plt.figure(figsize=(10, 6))
plt.plot(y_points, T_explicit_final, label='Explicit Euler', marker='o')
plt.plot(y_points, T_crank_final, label='Crank-Nicolson', marker='x')
plt.xlabel('y')
plt.ylabel('Temperature at x = 0.4')
plt.title('Vertical Temperature Profile at x = 0.4, t = 0.16')
plt.legend()
plt.grid(True)
plt.savefig("last_temperature_for_x04.png")




