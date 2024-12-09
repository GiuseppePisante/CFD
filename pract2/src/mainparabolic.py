import numpy as np
import matplotlib.pyplot as plt
from solverparabolic import solve_parabolic_1d
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline

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
plt.savefig("result1.png")


# Task 2.5
u_dim = u.T * 20
v_dim = v.T * 20

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot u
im1 = axs[0].imshow(u_dim, extent=[0, t_max * 0.0075, 0, length * 0.0075], aspect='auto', origin='lower', cmap='jet')
axs[0].set_title('Velocity distribution (u) over time')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
fig.colorbar(im1, ax=axs[0], label='Velocity')

# Plot v
im2 = axs[1].imshow(v_dim, extent=[0, t_max * 0.0075, 0, length * 0.0075], aspect='auto', origin='lower', cmap='jet')
axs[1].set_title('Velocity distribution (v) over time')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Position')
fig.colorbar(im2, ax=axs[1], label='Velocity')

plt.tight_layout()
plt.savefig("result2.png")

# task 2.6
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
u, v = solve_parabolic_1d(u0, v0, Re, dx, dt, t_max)

# Ensure X, Y grids match the dimensions of u_dim and v_dim
time_steps = u.shape[0]  # Number of time steps
positions = u.shape[1]   # Number of spatial positions
x = np.linspace(0, length, positions)
y = np.linspace(0, t_max, time_steps)
X, Y = np.meshgrid(x, y)

# Scale velocities to dimensional values
u_dim = u * 20  # Scale u to dimensional values
v_dim = v * 20  # Scale v to dimensional values

# Ensure the shapes match
assert u_dim.shape == X.shape, "Shape of u_dim does not match the grid"
assert v_dim.shape == X.shape, "Shape of v_dim does not match the grid"

# Compute the streamfunction
psi = np.zeros_like(u_dim)
for i in range(1, psi.shape[0]):  # Integrate along y-axis
    psi[i, :] = psi[i - 1, :] + u_dim[i, :] * dx

delta_99 = np.zeros_like(x)  # Initialize delta_99 to match x dimension
for i in range(len(x)):
    indices = np.where(u_dim[:, i] >= 0.99 * np.max(u_dim[:, i]))[0]  # Find y where u >= 0.99 * max(u) at each x
    if len(indices) > 0:
        delta_99[i] = y[indices[0]]  # First y position where condition is met
    else:
        delta_99[i] = np.nan  # If no index is found, use NaN for clarity

# Plot streamlines and streamfunction
fig, ax = plt.subplots(figsize=(10, 8))
strm = ax.streamplot(X, Y, u_dim, v_dim, color='r', linewidth=1.5, density=2)
contour = ax.contour(X, Y, psi, levels=20, cmap='coolwarm')
# Overlay the boundary layer edge (delta_99)
ax.plot(x, delta_99, 'k--', label='Boundary Layer Edge ($\delta_{99}$)')

# Labels and legend
ax.set_title('Streamlines and Contours of Streamfunction')
ax.set_xlabel('x (Position)')
ax.set_ylabel('y (Position)')
plt.colorbar(contour, label='Streamfunction')

plt.tight_layout()
plt.savefig("task_2_6_with_delta99.png")

# Task 2.7
# task 2.7: Velocity Profiles and Momentum Thickness
# Define specific x-positions to analyze velocity profiles
x_positions = [0.02, 0.04, 0.06]  # Example x-positions (can adjust based on domain)
x_indices = [np.argmin(np.abs(x - pos)) for pos in x_positions]  # Find indices corresponding to these positions

# Initialize momentum thickness array
momentum_thickness = np.zeros_like(x)

# Compute momentum thickness along x and extract velocity profiles
for i in range(len(x)):
    # Velocity profile at this x-position
    u_profile = u_dim[:, i]  # Extract u_dim at a specific x
    u_max = np.max(u_profile)  # Find max velocity at this x
    if u_max > 0:  # Ensure no division by zero
        u_ratio = u_profile / u_max  # Normalize velocity profile
        momentum_thickness[i] = np.trapz(u_ratio * (1 - u_ratio), y)  # Compute momentum thickness using trapezoidal rule
    else:
        momentum_thickness[i] = np.nan  # Assign NaN for undefined values

# Plot velocity profiles at specific x-positions
fig, ax1 = plt.subplots(figsize=(10, 8))
for idx, x_idx in enumerate(x_indices):
    ax1.plot(u_dim[:, x_idx], y, label=f'x = {x_positions[idx]:.3f}')

# Add labels and legend for velocity profiles
ax1.set_title('Velocity Profiles at Selected x-Positions')
ax1.set_xlabel('u (Velocity)')
ax1.set_ylabel('y (Position)')
ax1.legend()
plt.tight_layout()
plt.savefig("task_2_7_velocity_profiles.png")

# Plot momentum thickness along x
fig, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(x, momentum_thickness, 'b-', label='Momentum Thickness ($\\theta$)')

# Add labels and legend for momentum thickness
ax2.set_title('Momentum Thickness Along the Domain')
ax2.set_xlabel('x (Position)')
ax2.set_ylabel('Momentum Thickness ($\\theta$)')
ax2.legend()
plt.tight_layout()
plt.savefig("task_2_7_momentum_thickness.png")

# Integrate my functionality for streamlines and streamfunctions
X_fine, Y_fine = np.meshgrid(x, y)
u_interp = RectBivariateSpline(x, y, u_dim.T)
v_interp = RectBivariateSpline(x, y, v_dim.T)

def streamline_ode(x, y, u_interp, v_interp):
    """
    Computes the ODE for streamlines.
    
    Parameters:
        x: float
            Current x position.
        y: array-like
            Current y position(s).
        u_interp: RectBivariateSpline
            Interpolated u velocity field.
        v_interp: RectBivariateSpline
            Interpolated v velocity field.
    
    Returns:
        dydx: array-like
            Derivative of y with respect to x, i.e., v/u.
    """
    u = u_interp(x, y, grid=False)  # Interpolated u at (x, y)
    v = v_interp(x, y, grid=False)  # Interpolated v at (x, y)
    dydx = v / u  # Slope of streamline
    return dydx


# Generate streamlines using ODE-based method
start_points = np.linspace(0, np.max(y), 5)  # Starting y points
streamlines = []
for y0 in start_points:
    sol = solve_ivp(
        lambda x, y: streamline_ode(x, y, u_interp, v_interp),
        [0, length],
        [y0],
        dense_output=True
    )
    streamlines.append((sol.t, sol.y[0]))

plt.figure(figsize=(10, 8))
plt.contourf(X_fine, Y_fine, u_dim, levels=20, cmap='viridis')
for t, y_vals in streamlines:
    plt.plot(t, y_vals, 'r-')
plt.title("Streamlines Computed with ODE Integration")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="u (velocity)")
plt.tight_layout()
plt.savefig("task_streamlines_ODE.png")
