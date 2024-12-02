import numpy as np

def build_matrix(Nx, Ny, dx, dy, Re, u_initial, v_initial):

    N = 2 * Nx * Ny  
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # Costruzione della matrice A e del vettore b
    for j in range(Ny):      # Direzione y
        for i in range(Nx):  # Direzione x

            idx_u = j * Nx + i             # Indice per u(i,j)
            idx_v = Nx * Ny + j * Nx + i   # Indice per v(i,j)
            
            # Inner points
            if 0 < i < Nx-1 and 0 < j < Ny-1:

                # Continuity equation
                A[idx_u, idx_u] = 1 / dx
                A[idx_u, idx_u - 1] = -1 / dx

                A[idx_u, idx_v + Nx] = 1 / (2 * dy)
                A[idx_u, idx_v - Nx] = -1 / (2 * dy)
                
                # Momentum equation 
                A[idx_v, idx_u] = u_initial[j, i] / dx
                A[idx_v, idx_u - 1] = -u_initial[j, i] / dx

                A[idx_v, idx_u + Nx] = v_initial[j, i] / (2 * dy)
                A[idx_v, idx_u - Nx] = -v_initial[j, i] / (2 * dy)

                A[idx_v, idx_u + Nx] += 1 / (Re * dy**2)
                A[idx_v, idx_u] += 2 / (Re * dy**2)
                A[idx_v, idx_u - Nx] += -1 / (Re * dy**2)
            
            # Boundary points
            # Bordo sinistro (x=0)
            if i == 0:
                A[idx_u, idx_u] = 1 / dx
                A[idx_u, idx_u + 1] = -1 / dx

                A[idx_v, idx_u] = -u_initial[j, i] / dx
                A[idx_v, idx_u + 1] = u_initial[j, i] / dx

                b[idx_u] = 1  # Condizione di bordo u(0,y) = 1
                
            # Bordo destro (x=1)
            if i == Nx-1:
                b[idx_u] = 1  # Condizione di bordo u(1,y) = 1
                
            # Bordo inferiore (y=0)
            if j == 0:
                A[idx_u, idx_v] = 1 / (2 * dy)
                A[idx_u, idx_v - Nx] = -1 / (2 * dy)

                A[idx_v, idx_v] = v_initial[j, i] / (2 * dy)
                A[idx_v, idx_v - Nx] = -v_initial[j, i] / (2 * dy)

                A[idx_v, idx_u - 2*Nx] += 1 / (Re * dy**2)
                A[idx_v, idx_u - Nx] += 2 / (Re * dy**2)
                A[idx_v, idx_u] += -1 / (Re * dy**2)

            # Bordo superiore (y=infinito)
            if j == Ny-1:
                A[idx_u, idx_v + Nx] = 1 / (2 * dy)
                A[idx_u, idx_v] = -1 / (2 * dy)

                A[idx_v, idx_v + Nx] = v_initial[j, i] / (2 * dy)
                A[idx_v, idx_v] = -v_initial[j, i] / (2 * dy)

                A[idx_v, idx_u + 2*Nx] += 1 / (Re * dy**2)
                A[idx_v, idx_u - Nx] += 2 / (Re * dy**2)
                A[idx_v, idx_u] += -1 / (Re * dy**2)
                
            # Condizioni di bordo per v
            if i == 0 or i == Nx-1:  # Bordi sinistro e destro per v
                b[idx_v] = 0         # Condizione v(0,y) = 0 o v(1,y) = 0

            if j == Ny-1:            # Bordo superiore per v
                b[idx_v] = 0         # Condizione v(x,infinito) = 0
    
    return A, b