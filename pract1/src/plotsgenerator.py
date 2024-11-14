import numpy as np
import matplotlib.pyplot as plt
import mesh
import solver
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import time 

def heat_transfer_pde(T, t, x, y):
    # 2D heat equation: dT/dt = (d^2T/dx^2 + d^2T/dy^2)
    dTdx2 = solver.second_derivative(T, x[0, :], axis=1)
    dTdy2 = solver.second_derivative(T, y[:, 0], axis=0)

    dTdt = dTdx2 + dTdy2
    return dTdt

# To keep track of the time spent by each method, for task 1.3
explicit_euler_time = 0
crank_nicolson_time = 0

# Create the mesh
mesh_instance = mesh.Mesh(x_start=0, x_end=1, y_start=0, y_end=1, x_points=40, y_points=40)
mesh_instance.mesh_generator()
x, y = mesh_instance.x, mesh_instance.y

# Initial condition
T_initial = np.zeros_like(x)

# Time-interval parameters
t_start = 0
t_end = 0.16

# You can set num_time_steps = 16, 160, and 1600 in order to get a dt = 0.01, 0.001, and 0.0001
num_time_steps = 1600 

# Solve the PDE and store all time steps using Explicit-Euler method
dt = (t_end - t_start) / num_time_steps
T_explicit = T_initial.copy()
T_crank = T_initial.copy()
t = t_start

# Lists to store temperature evolution of the point (x, y) = (0.4, 0.4), for task 1.3
# Note: considering a spatial discretization with deltax = deltay = 1/40 we have that
# (x, y) = (0.4, 0.4) corresponds to the mesh coordinates of (16, 16), meaning (15, 15) in Python
T_explicit_point = [T_explicit[15,15]] 
T_crank_point = [T_crank[15,15]]

# Lists to store final temperature along the line x = 0.4, for task 1.3
T_explicit_final = []
T_crank_final = []

# Solve the problem with both methods and save the needed data
for _ in range(num_time_steps):
    start_time = time.time()
    T_explicit = solver.explicit_euler(heat_transfer_pde, T_explicit, x, y, t, dt)
    explicit_euler_time += time.time() - start_time  # Update cumulative time

    start_time = time.time()
    T_crank = solver.crank_nicolson(heat_transfer_pde, T_crank, x, y, t, dt)
    crank_nicolson_time += time.time() - start_time  # Update cumulative time
    t += dt
    T_explicit_point.append(T_explicit[15,15])
    T_crank_point.append(T_crank[15,15])

T_explicit_final = np.array(T_explicit[15, :])
T_crank_final = np.array(T_crank[15, :])

# Print the time
print(f"Total execution time for Explicit Euler: {explicit_euler_time:.4f} seconds")
print(f"Total execution time for Crank-Nicolson: {crank_nicolson_time:.4f} seconds")


# Create a time array based on dt and num_time_steps
time_points = np.linspace(t_start, t_end, num_time_steps + 1)  # +1 to include t=0

# Plot temperature evolution at (x, y) = (0.4, 0.4) for both methods
plt.figure(figsize=(10, 6))
plt.plot(time_points, T_explicit_point, label='Explicit Euler', marker='o')
plt.plot(time_points, T_crank_point, label='Crank-Nicolson', marker='x')
plt.xlabel('Time (t)')
plt.ylabel('Temperature at (x, y) = (0.4, 0.4)')
plt.title('Temperature Evolution at (x, y) = (0.4, 0.4)')
plt.legend()
plt.grid(True)
plt.savefig("temperature_evolution_of_a_point.png")


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
plt.savefig("final_temperature_along_a_line.png")




