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