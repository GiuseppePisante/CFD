import numpy as np
import matplotlib.pyplot as plt
import solver

# Parameters
Nx = 21
Ny = 21
time = 0.00005
dt = 0.00001
Re = 0.1

h = 1 / (Nx - 1)
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)

X, Y = np.meshgrid(x, y)

T = [0.02, 0.04, 0.06, 0.08, 0.1]

u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
p = np.zeros((Nx, Ny))

ut = np.zeros((Nx, Ny, 5))
vt = np.zeros((Nx, Ny, 5))
pt = np.zeros((Nx, Ny, 5))

time_counter = 1
iterations = 0

u[0, :] = 0.0
u[:, 0] = 0.0
u[:, -1] = 0.0
u[-1, :] = 1.0  # Top lid horizontal velocity
v[0, :] = 0.0
v[:, 0] = 0.0
v[:, -1] = 0.0
v[-1, :] = 0.0

un1 = u
vn1 = v

for t in np.arange(0, time, dt):
    print(t)
    d_u__d_x = solver.central_difference_x(u, h)
    d_u__d_y = solver.central_difference_y(u, h)
    d_v__d_x = solver.central_difference_x(v, h)
    d_v__d_y = solver.central_difference_y(v, h)
    laplace__u = solver.laplace(u, h)
    laplace__v = solver.laplace(v, h)

    # Solving momentum equations ignoring the pressure gradient
    u_star = u + dt * (- (u * d_u__d_x + v * d_u__d_y) + (1 / Re) * laplace__u)
    v_star = v + dt * (- (u * d_v__d_x + v * d_v__d_y) + (1 / Re) * laplace__v)

    d_u_star__d_x = solver.central_difference_x(u_star, h)
    d_v_star__d_y = solver.central_difference_y(v_star, h)

    # Solving Poisson's equation to predict pressure for current time step
    rhs = (1 / dt) * (d_u_star__d_x + d_v_star__d_y)
    p = solver.poisson(p, rhs, h)

    d_p__d_x = solver.central_difference_x(p, h)
    d_p__d_y = solver.central_difference_y(p, h)

    # Updating velocities based on predicted pressure
    un1 = u_star - (dt * d_p__d_x)
    vn1 = v_star - (dt * d_p__d_y)

    u = un1
    v = vn1

    if np.isclose(t, T[time_counter - 1], atol=dt):
        ut[:, :, time_counter - 1] = u  # Storing the results for each time step
        vt[:, :, time_counter - 1] = v
        pt[:, :, time_counter - 1] = p
        time_counter += 1

# Compute Residuals for Task 3.5
residual_x = np.zeros((Ny, Nx))
residual_y = np.zeros((Ny, Nx))
continuity_residual = np.zeros((Ny, Nx))

for j in range(1, Ny-1):
    for i in range(1, Nx-1):
        convection_x = ((u[j, i+1] * u[j, i+1] - u[j, i-1] * u[j, i-1]) / (2 * h) +
                        (u[j+1, i] * v[j+1, i] - u[j-1, i] * v[j-1, i]) / (2 * h))
        diffusion_x = (1/Re) * ((u[j, i+1] - 2*u[j, i] + u[j, i-1]) / h**2 +
                                (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / h**2)
        pressure_gradient_x = (p[j, i+1] - p[j, i-1]) / (2 * h)
        
        residual_x[j, i] = convection_x - diffusion_x + pressure_gradient_x

        convection_y = ((u[j, i+1] * v[j, i+1] - u[j, i-1] * v[j, i-1]) / (2 * h) +
                        (v[j+1, i] * v[j+1, i] - v[j-1, i] * v[j-1, i]) / (2 * h))
        diffusion_y = (1/Re) * ((v[j, i+1] - 2*v[j, i] + v[j, i-1]) / h**2 +
                                (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / h**2)
        pressure_gradient_y = (p[j+1, i] - p[j-1, i]) / (2 * h)
        
        residual_y[j, i] = convection_y - diffusion_y + pressure_gradient_y

        continuity_residual[j, i] = ((u[j, i+1] - u[j, i-1]) / (2 * h) +
                                     (v[j+1, i] - v[j-1, i]) / (2 * h))

# Plot the Residuals
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

c1 = axes[0].pcolormesh(X, Y, residual_x, shading='auto', cmap='viridis')
fig.colorbar(c1, ax=axes[0])
axes[0].set_title('X-Momentum Residual')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

c2 = axes[1].pcolormesh(X, Y, residual_y, shading='auto', cmap='viridis')
fig.colorbar(c2, ax=axes[1])
axes[1].set_title('Y-Momentum Residual')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

c3 = axes[2].pcolormesh(X, Y, continuity_residual, shading='auto', cmap='viridis')
fig.colorbar(c3, ax=axes[2])
axes[2].set_title('Continuity Residual')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')

plt.tight_layout()
plt.show()

# Existing Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
c = ax.contourf(X, Y, u, cmap='jet')
fig.colorbar(c, ax=ax)
ax.set_title('u at final time step')

ax = axes[1]
c = ax.contourf(X, Y, v, cmap='jet')
fig.colorbar(c, ax=ax)
ax.set_title('v at final time step')

ax = axes[0]
strm = ax.streamplot(X, Y, u, v)
ax.set_title('Streamlines at final time step')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.set_aspect('equal')

ax = axes[2]
c = ax.contourf(X, Y, p, cmap='jet')
fig.colorbar(c, ax=ax)
ax.set_title('p at final time step')

plt.tight_layout()
plt.show()
