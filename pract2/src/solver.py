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

        idx_u = np.arange(Nx * Ny)
        idx_v = Nx * Ny + np.arange(Nx * Ny)

        musk_inner = (idx_u % Nx != 0) & (idx_u % Nx != Nx-1) & (idx_u < Nx * Ny - Nx) & (idx_u >= Nx)
        musk_left = (idx_u % Nx == 0)
        musk_right = (idx_u % Nx == Nx-1)
        musk_bottom = (idx_u < Nx)
        musk_top = (idx_u >= Nx * Ny - Nx)

        # Continuity equation
        self.A[idx_u[musk_inner], idx_u[musk_inner]] += 1 / dx
        self.A[idx_u[1:][musk_inner[1:]], idx_u[:-1][musk_inner[1:]]] -= 1 / dx

        self.A[idx_u[Nx:][musk_inner[Nx:]], idx_v[:-Nx][musk_inner[:-Nx]]] += 1 / (2 * dy)
        self.A[idx_u[:-Nx][musk_inner[:-Nx]], idx_v[Nx:][musk_inner[:-Nx]]] -= 1 / (2 * dy)

        # Momentum equation
        self.A[idx_v[musk_inner], idx_u[musk_inner]] += 1 / dx
        self.A[idx_v[musk_inner], idx_u[musk_inner]] += 2 / (Re * dy**2)

        self.A[idx_v[:-Nx][musk_inner[:-Nx]], idx_u[Nx:][musk_inner[Nx:]]] -= 1 / (2 * dy)
        self.A[idx_v[:-Nx][musk_inner[:-Nx]], idx_u[Nx:][musk_inner[Nx:]]] -= 1 / (Re * dy**2)

        self.A[idx_v[Nx:][musk_inner[Nx:]], idx_u[:-Nx][musk_inner[:-Nx]]-Nx] += 1 / (2 * dy)
        self.A[idx_v[Nx:][musk_inner[Nx:]], idx_u[:-Nx][musk_inner[:-Nx]]-Nx] -= 1 / (Re * dy**2)

        self.A[idx_v[:-1][musk_inner[:-1]], idx_u[1:][musk_inner[1:]]-1] -= 1 / dx

        # Boundary points
        # Left boundary (x = 0)
        self.A[idx_u[musk_left], idx_u[musk_left]] = 1
        self.A[idx_v[musk_left], idx_v[musk_left]] = 1
        self.b[idx_u[musk_left]] = 1  # Free-stream inflow for u
        self.b[idx_v[musk_left]] = 0

        # Right boundary (x = Nx-1) - Outflow
        self.A[idx_u[musk_right], idx_u[musk_right]] = 1
        self.A[idx_v[musk_right], idx_v[musk_right]] = 1
        self.b[idx_u[musk_right]] = 1  # Free-stream outflow for u
        self.b[idx_v[musk_right]] = 0

        # Bottom boundary (y = 0) - No-slip wall
        self.A[idx_u[musk_bottom], idx_u[musk_bottom]] = 1
        self.A[idx_v[musk_bottom], idx_v[musk_bottom]] = 1
        self.b[idx_u[musk_bottom]] = 0  # u = 0
        self.b[idx_v[musk_bottom]] = 0  # v = 0

        # Top boundary (y = Ny-1) - Far-field
        self.A[idx_u[musk_top], idx_u[musk_top]] = 1
        self.A[idx_v[musk_top], idx_v[musk_top]] = 1
        self.b[idx_u[musk_top]] = 1  # Free-stream u velocity
        self.b[idx_v[musk_top]] = 0  # No v velocity


    def updateMatrix(self, solution):
        Nx = self.Nx
        Ny = self.Ny
        dx = self.dx
        dy = self.dy
        Re = self.Re

        idx_u = np.arange(Nx * Ny)
        idx_v = Nx * Ny + np.arange(Nx * Ny)

        musk_inner = (idx_u % Nx != 0) & (idx_u % Nx != Nx-1) & (idx_u < Nx * Ny - Nx) & (idx_u >= Nx)
        
        # Momentum equation 
        self.A[idx_v[musk_inner], idx_u[musk_inner]] = solution[idx_u][musk_inner] / dx
        self.A[idx_v[musk_inner], idx_u[musk_inner]] += 2 / (Re * dy**2)

        self.A[idx_v[:-Nx][musk_inner[:-Nx]], idx_u[Nx:][musk_inner[Nx:]]] = - solution[idx_v[:-Nx][musk_inner[:-Nx]]] / (2 * dy)
        self.A[idx_v[:-Nx][musk_inner[:-Nx]], idx_u[Nx:][musk_inner[Nx:]]] -= 1 / (Re * dy**2)

        self.A[idx_v[Nx:][musk_inner[Nx:]], idx_u[:-Nx][musk_inner[:-Nx]]-Nx] = solution[idx_v[Nx:][musk_inner[Nx:]]] / (2 * dy)
        self.A[idx_v[Nx:][musk_inner[Nx:]], idx_u[:-Nx][musk_inner[:-Nx]]-Nx] -= 1 / (Re * dy**2)

        self.A[idx_v[:-1][musk_inner[:-1]], idx_u[1:][musk_inner[1:]]-1] = - solution[idx_u[:-1][musk_inner[:-1]]] / dx
