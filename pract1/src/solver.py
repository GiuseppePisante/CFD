import numpy as np

def second_derivative(u, coord, axis):
    du2 = np.zeros_like(u)
    h = coord[1] - coord[0]
    if axis == 0:
        du2[1:-1, :] = (u[:-2, :] - 2 * u[1:-1, :] + u[2:, :]) / h**2
    elif axis == 1:
        du2[:, 1:-1] = (u[:, :-2] - 2 * u[:, 1:-1] + u[:, 2:]) / h**2
    return du2

def explicit_euler(pde_func, u_initial, x, y, t, dt):
    u = u_initial.copy()
    
    # Boundary conditions
    u[0, :] = 1 - y[:,0]**3
    u[-1, :] = 1 - np.sin(np.pi * y[:,0] / 2)
    u[:, 0] = 1
    u[:, -1] = 0

    u += dt * pde_func(u, t, x, y)

    return u

def crank_nicolson(pde_func, u_initial, x, y, t, dt):
    u = u_initial.copy()
    nx, ny = u.shape
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    
    # Boundary conditions
    u[0, :] = 1 - y[:,0]**3
    u[-1, :] = 1 - np.sin(np.pi * y[:,0] / 2)
    u[:, 0] = 1
    u[:, -1] = 0
    
    # Construct the pentadiagonal matrix
    A = np.zeros((nx * ny, nx * ny))
    b = np.zeros(nx * ny)
    
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                # Boundary points
                A[idx, idx] = 1
                b[idx] = u[i, j]
            else:
                A[idx, idx] = 1 + dt * (1 / dx**2 + 1 / dy**2)
                if i > 0:
                    A[idx, idx - ny] = -dt / (2 * dx**2)
                if i < nx - 1:
                    A[idx, idx + ny] = -dt / (2 * dx**2)
                if j > 0:
                    A[idx, idx - 1] = -dt / (2 * dy**2)
                if j < ny - 1:
                    A[idx, idx + 1] = -dt / (2 * dy**2)
                b[idx] = u[i, j] + dt * pde_func(u, t, x, y)[i, j]
    
    # Time-stepping loop
    u_new_flat = np.linalg.solve(A, b)
    u_new = u_new_flat.reshape((nx, ny))
    
    return u_new