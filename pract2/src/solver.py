import numpy as np

class LinearSystem():

    def __init__(self, Nx, Ny, dx, dy, Re):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.Re = Re
        self.N = 2 * Nx * Ny
          
        self.A = np.zeros((self.N, self.N))
        self.b = np.zeros(self.N)
    
    def build_matrix(self):
        Nx = self.Nx
        Ny = self.Ny
        dx = self.dx
        dy = self.dy
        Re = self.Re
        
        # Costruzione della matrice A e del vettore b
        for j in range(Nx):      # Direzione y
            for i in range(Ny):  # Direzione x

                idx_u = j * Nx + i             # Indice per u(i,j)
                idx_v = Nx * Ny + j * Nx + i   # Indice per v(i,j)
                
                # Inner points
                if 0 < i < Ny-1 and 0 < j < Nx-1:

                    # Continuity equation
                    self.A[idx_u, idx_u] += 1 / dx
                    self.A[idx_u, idx_u - Nx] -= 1 / dx

                    self.A[idx_u, idx_v + 1] += 1 / (2 * dy)
                    self.A[idx_u, idx_v - 1] -= 1 / (2 * dy)
                    
                    # Momentum equation 
                    self.A[idx_v, idx_u] += 1 / dx
                    self.A[idx_v, idx_u] += 2 / (Re * dy**2)

                    self.A[idx_v, idx_u - 1] -= 1 / (2 * dy)
                    self.A[idx_v, idx_u - 1] += 1 / (Re * dy**2)


                    self.A[idx_v, idx_u + 1] = 1 / (2 * dy)
                    self.A[idx_v, idx_u + 1] += 1 / (Re * dy**2)

                    self.A[idx_v, idx_u - Nx] -= 1 / dx
                
                # Boundary points
                # Left boundary (x = 0)
                if i == 0:
                    self.A[idx_u, idx_u] = 1
                    self.A[idx_v, idx_v] = 1
                    self.b[idx_u] = 1  # Free-stream inflow for u
                    self.b[idx_v] = 0  # No v velocity

                # Right boundary (x = Nx-1) - Outflow
                if i == Nx-1:
                    # self.A[idx_u, idx_u] = 1
                    self.A[idx_u, idx_u] = 1
                    self.A[idx_v, idx_v] = 1
                    self.b[idx_u] = 1  # Free-stream outflow for u
                    self.b[idx_v] = 0  # No v velocity

                # Bottom boundary (y = 0) - No-slip wall
                if j == 0:
                    self.A[idx_u, idx_u] = 1
                    self.A[idx_v, idx_v] = 1
                    self.b[idx_u] = 0  # u = 0
                    self.b[idx_v] = 0  # v = 0

                # Top boundary (y = Ny-1) - Far-field
                if j == Ny-1:
                    self.A[idx_u, idx_u] = 1
                    self.A[idx_v, idx_v] = 1
                    self.b[idx_u] = 1  # Free-stream u velocity
                    self.b[idx_v] = 0  # No v velocity

        print(np.count_nonzero(self.A[Nx*Ny +19, :]))


    def updateMatrix(self, u_initial, v_initial):
        Nx = self.Nx
        Ny = self.Ny
        dx = self.dx
        dy = self.dy
        Re = self.Re

        for j in range(Nx):      # Direzione y
            for i in range(Ny):  # Direzione x

                idx_u = j * Nx + i             # Indice per u(i,j)
                idx_v = Nx * Ny + j * Nx + i   # Indice per v(i,j)
                
                # Inner points
                if 0 < i < Nx-1 and 0 < j < Ny-1:
                    
                    # Momentum equation 
                    self.A[idx_v, idx_u] = u_initial[i,j] / dx
                    self.A[idx_v, idx_u] -= 2 / (Re * dy**2)

                    self.A[idx_v, idx_u - 1] = v_initial[i,j] / (2 * dy)
                    self.A[idx_v, idx_u - 1] += 1 / (Re * dy**2)

                    self.A[idx_v, idx_u + 1] = - v_initial[i,j] / (2 * dy)
                    self.A[idx_v, idx_u + 1] += 1 / (Re * dy**2)

                    self.A[idx_v, idx_u - Nx] = - u_initial[i,j] / dx