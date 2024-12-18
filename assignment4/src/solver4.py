import numpy as np

def max_change(phi_old, phi_new):
        return np.max(np.abs(phi_new - phi_old))

def CDS(phi, dt, dx, Pe, nt_max, tol=1e-6):

    phi_new = np.zeros_like(phi)

    for n in range(nt_max):
        phi_old = phi.copy()

        for i in range(1, len(phi) - 1):
            advection_term = dt / (2 * dx) * (phi[i + 1] - phi[i - 1])
            diffusion_term = dt / (Pe * dx**2) * (phi[i + 1] - 2 * phi[i] + phi[i - 1])
            phi_new[i] = phi[i] - advection_term + diffusion_term

        phi_new[0] = 0.0
        phi_new[-1] = 1.0

        if max_change(phi, phi_new) < tol:
            break

        phi[:] = phi_new

    return phi

def UP1(phi, dt, dx, Pe, nt_max, tol=1e-6):

    phi_new = np.zeros_like(phi)

    for n in range(nt_max):
        phi_old = phi.copy()

        for i in range(1, len(phi) - 1):
            advection_term = dt / dx * (phi[i] - phi[i - 1])
            diffusion_term = dt / (Pe * dx**2) * (phi[i + 1] - 2 * phi[i] + phi[i - 1])
            phi_new[i] = phi[i] - advection_term + diffusion_term

        phi_new[0] = 0.0
        phi_new[-1] = 1.0

        if max_change(phi, phi_new) < tol:
            break

        phi[:] = phi_new

    return phi

def UP2(phi, dt, dx, Pe, nt_max, tol=1e-6):

    phi_new = np.zeros_like(phi)

    for n in range(nt_max):
        phi_old = phi.copy()

        for i in range(1, len(phi) - 1):
            advection_term = dt / (2 * dx) * (3 * phi[i] - 4 * phi[i - 1] + phi[i - 2])
            diffusion_term = dt / (Pe * dx**2) * (phi[i + 1] - 2 * phi[i] + phi[i - 1])
            phi_new[i] = phi[i] - advection_term + diffusion_term

        phi_new[0] = 0.0
        phi_new[1] = 0.0
        phi_new[-1] = 1.0

        if max_change(phi, phi_new) < tol:
            break

        phi[:] = phi_new

    return phi