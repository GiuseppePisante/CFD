import numpy as np
from solverparabolic import explicit_euler_solver, apply_boundary_conditions, initialize_conditions, plot_solution_2d

# Example usage
if __name__ == "__main__":
    L = 1.0
    Nx, Ny = 10, 10
    Re = 10000.0

    delta_99 = 4.91 / np.sqrt(Re) * L
    H = 2.0 * delta_99
    dx, dy = H / Nx, L / Ny

    dt = 0.0001
    x_final = 1.0
    y_points = Ny + 1
    x_points = Nx + 1
    y = np.linspace(0, H, y_points)
    x = np.linspace(0, L, x_points)

    u_initial, v_initial = initialize_conditions(y_points)
    u_initial = apply_boundary_conditions(u_initial,2)
    v_initial = apply_boundary_conditions(v_initial,2)
    u_initial[0]=1

    u = np.zeros((x_points, y_points))
    v = np.zeros((x_points, y_points))

    u[0, :] = u_initial
    v[0, :] = v_initial

    for i in range(1, x_points):
        u[i, :], v[i, :] = explicit_euler_solver(u[i-1, :],v[i-1,:], dy, dt, dt,  Re)

    print(u)
    plot_solution_2d(x, y, u, v)
