import numpy as np

def solve_CDS(phi, Dt, Dx, Pe, nt_max, tol=1e-6):

    phi_new = np.zeros_like(phi)
    time = 0.0
    steady_state = False

    def max_change(phi_old, phi_new):
        return np.max(np.abs(phi_new - phi_old))

    for n in range(nt_max):
        phi_old = phi.copy()

        # Update interior points using the scheme
        for i in range(1, len(phi) - 1):
            phi_new[i] = ( phi[i] - 0.5 * Dt / Dx * (phi[i + 1] - phi[i - 1]) + Dt / (Pe * Dx**2) * (phi[i + 1] - 2 * phi[i] + phi[i - 1]))

        # Enforce boundary conditions
        phi_new[0] = 0.0
        phi_new[-1] = 1.0

        # Check for steady state
        if max_change(phi, phi_new) < tol:
            steady_state = True
            break

        phi[:] = phi_new
        time += Dt

    return phi, time, n + 1, steady_state