import numpy as np

def solve_parabolic_1d(u0,v0, Re, dy, dt, t_max):
    
    nx = len(u0)
    nt = int(t_max / dt)
    u = np.zeros((nt, nx))
    v = np.zeros((nt, nx))
    u[0, :] = u0
    v[0, :] = v0

    # Boundary conditions
    u[0, 0] = 0 
    u[0, -1] = 1
    v[0, 0] = 0

    for n in range(1, nt):
        v[n, 1] = v[n-1, 0]/dy + (u[n, 1]-u[n-1, 1])/dt
        for i in range(1, nx-1):
            u[n, i] = u[n-1, i] + dt / u[n-1, i] * ((u[n-1, i+1] - 2*u[n-1, i] + u[n-1, i-1])/(Re * dy ** 2))
            # - v[n-1,i] * (u[n-1,i + 1] - u[n-1,i - 1]) / (2 * dy)
            v[n, i + 1] = v[n-1,i - 1]/(2 * dy) + (u[n-1,i]-u[n,i])/dt
        # Boundary conditions
        u[n, 0] = 0
        u[n, -1] = 1
        v[n, 0] = 0

    return u, v
