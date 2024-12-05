import numpy as np
import matplotlib.pyplot as plt

def explicit_euler_solver(u_initial, v_initial, dy, dt, t_final, Re):
    u = u_initial.copy()
    v=v_initial.copy()
    n = len(u)
    t = 0

    # Stability condition for explicit Euler method
    if dt > dy**2 / 2:
        raise ValueError("Time step dt is too large for stability. Reduce dt or increase dy.")

    while t < t_final:
        
        u_new = u.copy()
        v_new = v.copy()
        for i in range(1, n - 1):
            u_new[i] = u[i] + dt / u[i] * ((u[i + 1] - 2 * u[i] + u[i - 1]) / (Re * dy ** 2) - v[i] * (u[i + 1] - u[i - 1]) / (2 * dy))
            v_new[i] = v[i - 1]/dy + (u_new[i]-u[i])/dt
        u_new=apply_boundary_conditions(u_new,1)
        v_new=apply_boundary_conditions(u_new,2)
        
        u = u_new
        v = v_new
        t += dt


    return u,v

def apply_boundary_conditions(u,flag):
    u[0] = 0.0  # Boundary condition at y = 0
    if(flag==1):
        u[-1] = 1.0
    return u

def initialize_conditions(y_points):
    u_initial = np.ones(y_points)
    v_initial = np.zeros(y_points)
    return u_initial, v_initial

def plot_solution_2d(x, y, u, v):
    X, Y = np.meshgrid(x, y)
    plt.figure()
    plt.contourf(X, Y, u.T, cmap='viridis')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('u velocity in 2D domain')
    plt.show()

    plt.figure()
    plt.contourf(X, Y, v.T, cmap='viridis')
    plt.colorbar(label='v')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('v velocity in 2D domain')
    plt.show()