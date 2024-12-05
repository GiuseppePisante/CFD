import numpy as np
import matplotlib.pyplot as plt
from solverparabolic import solve_parabolic_1d

# Parameters
Re = 10000  # Diffusivity
dx = 0.001      # Spatial step size
dt = 0.0001     # Time step size
t_max = 1.0   # Maximum time
length = 0.08  # Length of the rod

# Initial condition
nx = int(length / dx) + 1
x = np.linspace(0, length, nx)
u0 = np.ones(nx)  # Initial condition set to 1 everywhere
v0 = np.zeros(nx)  # Initial velocity distribution
# Solve the heat equation
u, v = solve_parabolic_1d(u0,v0, Re, dx, dt, t_max)

# Plot the results as a 2D heatmap
plt.imshow(u.T, extent=[0, t_max, 0, length], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Velocity')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Velocity distribution over time')
plt.show()

plt.imshow(v.T, extent=[0, t_max, 0, length], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Velocity')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Velocity distribution over time')
plt.show()

# Plot the value of a point u over time
point_index = nx // 2  # Choose the middle point
time = np.linspace(0, t_max, int(t_max / dt))
plt.plot(time, u[:, point_index])
plt.xlabel('Time')
plt.ylabel('Velocity at midpoint')
plt.title('Velocity at midpoint over time')
plt.show()

