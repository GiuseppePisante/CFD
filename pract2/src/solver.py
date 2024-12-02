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
                # Bordo sinistro (x=0)
                if i == 0:
                    self.A[idx_u, idx_u] = 1 
                    self.A[idx_v, idx_u] = 1
                    self.A[idx_v, idx_u] = 1

                    self.b[idx_u] = 1  # Condizione di bordo u(0,y) = 1
                    self.b[idx_v] = 0  # Condizione v(0,y)
                    
                # Bordo destro (x=1)
                if i == Ny-1:
                    self.A[idx_u, idx_u] = 1 
                    self.A[idx_v, idx_u] = 1
                    self.A[idx_v, idx_u] = 1
                    self.b[idx_u] = 1  # Condizione di bordo u(1,y) = 1
                    self.b[idx_v] = 0  # Condizione v(1,y)
                    
                # Bordo superiore
                if j == 0:
                    self.A[idx_u, idx_v] = 1 
                    self.A[idx_v, idx_u] = 1
                    self.A[idx_u, idx_u] = 1

                    self.b[idx_u] = 1
                    self.b[idx_v] = 0


                # Bordo inferiore
                if j == Nx-1:
                    self.A[idx_u, idx_u] = 1 
                    self.A[idx_v, idx_u] = 1
                    self.b[idx_v] = 0  # v(1,y) = 0
                    self.b[idx_u] = 0  # v(1,y) = 0


    def updateMatrix(self, u_initial, v_initial):
        Nx = self.Nx
        Ny = self.Ny
        dx = self.dx
        dy = self.dy
        Re = self.Re

        for j in range(Ny):      # Direzione y
            for i in range(Nx):  # Direzione x

                idx_u = j * Nx + i             # Indice per u(i,j)
                idx_v = Nx * Ny + j * Nx + i   # Indice per v(i,j)
                
                # Inner points
                if 0 < i < Nx-1 and 0 < j < Ny-1:
                    
                    # Momentum equation 
                    self.A[idx_v, idx_u] = u_initial[i,j] / dx
                    self.A[idx_v, idx_u] -= 2 / (Re * dy**2)

                    self.A[idx_v, idx_u - 1] = v_initial[i,j] / (2 * dy)
                    self.A[idx_v, idx_u - 1] += 1 / (Re * dy**2)


                    self.A[idx_v, idx_u + 1] = -v_initial[i,j] / (2 * dy)
                    self.A[idx_v, idx_u + 1] += 1 / (Re * dy**2)


                    self.A[idx_v, idx_u - Nx] = - u_initial[i,j] / dx
                
                # Boundary points
                # Bordo sinistro (x=0)
                if i == 0:
                    """ self.A[idx_v, idx_u] -= u_initial[i,j] / dx
                    self.A[idx_v, idx_u + Nx] += u_initial[i,j] / dx """
                    self.A[idx_v, idx_u] = 1
                # Bordo superiore
                if j == 0:
                    """ self.A[idx_v, idx_u] = -v_initial[i,j] / dy
                    self.A[idx_v, idx_u] +=  1 / (Re * dy**2) """
                    self.A[idx_v, idx_u] = 1

                    self.A[idx_v, idx_u + 1] = v_initial[i,j] / dy
                    self.A[idx_v, idx_u + 1] += 2 / (Re * dy**2)

                # Bordo inferiore
                if j == Ny-1:
                    self.A[idx_v, idx_u] = u_initial[i,j] / dx
                    self.A[idx_v, idx_u] += v_initial[i,j] / dy
                    self.A[idx_v, idx_u] -= 1 / (Re * dy**2)

                    self.A[idx_v, idx_u - 2] = 1 / (Re * dy**2)

                    self.A[idx_v, idx_u - 1] = 2 / (Re * dy**2)
                    self.A[idx_v, idx_u - 1] -= v_initial[i,j] / dy