import numpy as np

def second_derivative(u, coord, axis):
    return np.gradient(np.gradient(u, coord, axis=axis), coord, axis=axis)

def explicit_euler(pde_func, u_initial, x, y, t_start, t_end, num_time_steps):

    dt = (t_end - t_start) / num_time_steps
    u = u_initial.copy()
    t = t_start

    for _ in range(num_time_steps):
        u += dt * pde_func(u, t, x, y)
        t += dt

    return u

def Crank_Nicolson(u_initial, x, y, t_start, t_end, num_time_steps, alpha=0.01):

    dt = (t_end - t_start) / num_time_steps
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    u = u_initial.copy()
    
    nx, ny = u.shape
    
    # Coefficients
    rx = dt / (2 * dx**2)
    ry = dt / (2 * dy**2)
    
    # Create tridiagonal matrices Ax and Ay
    Ax = np.diag((1 + 2 * rx) * np.ones(nx)) + np.diag(-rx * np.ones(nx - 1), k=1) + np.diag(-rx * np.ones(nx - 1), k=-1)
    Ay = np.diag((1 + 2 * ry) * np.ones(ny)) + np.diag(-ry * np.ones(ny - 1), k=1) + np.diag(-ry * np.ones(ny - 1), k=-1)
    
    for _ in range(num_time_steps):
        # Apply boundary conditions
        u = u_initial.copy()
        
        # Solve the system
        u_half = np.linalg.solve(Ax, u @ Ay.T)
        u = np.linalg.solve(Ay, u_half.T @ Ax.T).T
    
    return u