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

# Plot the results as 2D heatmaps in a single figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot u
im1 = axs[0].imshow(u.T, extent=[0, t_max, 0, length], aspect='auto', origin='lower', cmap='jet')
axs[0].set_title('Velocity distribution (u) over time')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
fig.colorbar(im1, ax=axs[0], label='Velocity')

# Plot v
im2 = axs[1].imshow(v.T, extent=[0, t_max, 0, length], aspect='auto', origin='lower', cmap='jet')
axs[1].set_title('Velocity distribution (v) over time')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Position')
fig.colorbar(im2, ax=axs[1], label='Velocity')

plt.tight_layout()
plt.show()
