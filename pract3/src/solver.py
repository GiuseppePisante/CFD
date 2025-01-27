import numpy as np

def simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_n,d_e,N,nu,b,h):

    # Under-relaxation factors
    alpha = 0.65
    alpha_p = 0.35
    err_u, err_v, err_p =1, 1, 1
    it=0
    itermax=100
    tol = 1e-2

    while ((err_u > tol) & (err_v > tol) & (err_p > tol) &(it<itermax)):

        for i in range(1, N-1):  # Loop over internal cells
            for j in range(1, N-1):
                # Interpolated face velocities
                u_E = 0.5*(u[i,j] + u[i,j+1])
                u_W = 0.5*(u[i,j] + u[i,j-1])
                v_N = 0.5*(v[i,j+1] + v[i,j])
                v_S = 0.5*(v[i,j-1] + v[i,j])

                # Coefficients (Convective and Diffusive)
                a_C = 0.5 * u_E * h - 0.5 * u_W * h + 0.5 * v_S * h - 0.5 * v_N * h - 2* nu
                a_N= -0.5 * v_N * h + nu
                a_S = 0.5 * v_S * h + nu
                a_E = - 0.5 * u_W * h - nu
                a_W = 0.5 * u_E * h - nu

                d_e[i,j] = -h/a_C

                # Momentum equation (Intermediate u*)
                u_star[i,j] = (a_E * u[i,j+1] + a_W * u[i,j-1] + a_S * u[i-1,j] + a_N * u[i+1,j] + h * (p[i,j+1] - p[i,j])) / a_C

        for i in range(1, N-1):  # Loop over internal cells
            for j in range(1, N-1):
                # Interpolated face velocities
                u_E = 0.5*(u[i,j] + u[i,j+1])
                u_W = 0.5*(u[i,j] + u[i,j-1])
                v_N = 0.5*(v[i,j+1] + v[i,j])
                v_S = 0.5*(v[i,j-1] + v[i,j])

                # Coefficients (Convective and Diffusive)
                a_C = 0.5 * v_S * h - 0.5 * v_N * h + 0.5 * u_E * h - 0.5 * u_W * h - 2* nu
                a_N= -0.5 * v_N * h + nu
                a_S = 0.5 * v_S * h + nu
                a_E= 0.5 * u_E * h - nu
                a_W = -0.5 * u_W * h - nu
                
                d_n[i,j] = -h/a_C

                # Momentum equation (Intermediate u*)
                v_star[i,j]= (a_E * v[i,j+1] + a_W * v[i,j-1] + a_S * v[i-1,j] + a_N * v[i+1,j] + h * (p[i+1,j] - p[i,j])) / a_C

        # Boundary conditions
        u_star[0, :] = u_star[1,:]/3
        u_star[N - 1, :] = 2/3 - u_star[N - 2,:]/3
        u_star[0 : N -2,0] = 0
        u_star[1 : N - 2, N] = 0

        # Boundary Conditions for v
        v_star[1 : N - 1, 0] = v_star[1 : N - 1, 1]/3  # Left wall
        v_star[1 : N - 1, N - 1] = v_star[1 : N - 1, N - 2]/3  # Right wall
        v_star[0, :] = 0  # Top wall
        v_star[N, :] = 0  # Bottom wall

# FINO A QUA
        # Correction term (Initialization)
        pc[0:N, 0:N] = 0
        
        #Continuity equation a.k.a. pressure correction - Interior
        for i in range(1,N):
            for j in range(1,N):
                a_E = -d_e[i,j]*h
                a_W = -d_e[i,j-1]*h
                a_N = -d_n[i-1,j]*h
                a_S = -d_n[i,j]*h
                a_P = a_E + a_W + a_N + a_S
                b[i,j] = -(u_star[i,j] - u_star[i,j-1])*h + (v_star[i,j] - v_star[i-1,j])*h

                pc[i,j] = (a_E*pc[i,j+1] + a_W*pc[i,j-1] + a_N*pc[i-1,j] + a_S*pc[i+1,j] + b[i,j])/a_P
        
        
        #Correcting the pressure field
        for i in range( 1,N):
            for j in range( 1,N) :
                p_new[i,j] = p[i,j] + alpha_p*pc[i,j]


        # Pressure field Correction
        p_new[1:N, 1:N] = p[1:N, 1:N] + alpha_p * pc[1:N, 1:N]

        # Boundary Continuity
        p_new[0,:] = p_new[1,:]
        p_new[N,:] = p_new[N-1,:]
        p_new[:,0] = p_new[:,1]
        p_new[:,N]= p_new[:,N-1]

        # Velocity Correction
        u_new[1:N, 1:N-1] = u_star[1:N, 1:N-1] + d_e[1:N, 1:N-1] * (pc[1:N, 2:N] - pc[1:N, 1:N-1])
        v_new[1:N-1, 1:N] = v_star[1:N-1, 1:N] + d_n[1:N-1, 1:N] * (pc[2:N, 1:N] - pc[1:N-1, 1:N])



        # Boundary x-correction
        u_new[0,:] = 2 - u_new[1,:]
        u_new[N,:] = -u_new[N-1,:]
        u_new[1:N-1,0] = 0
        u_new[1:N-1,N-1] = 0

        v_new[1:N-1, 1:N] = v_star[1:N-1, 1:N] + d_n[1:N-1, 1:N] * (pc[1:N-1, 1:N] - pc[2:N, 1:N])


        # Boundary y-momentum
        v_new[:,0] = -v_new[:,1]
        v_new[:,N] = -v_new[:,N-1]
        v_new[0,1:N-1] = 0
        v_new[N-1,1:N-1] = 0

        
        # Residual computation
        err_u = np.abs(u_new - u).max() 
        err_v = np.abs(v_new - v).max()
        err_p = np.abs(p_new - p).max()
        u = u_new.copy()
        v = v_new.copy()
        p = p_new.copy()
        it = it + 1

    return u,v,p



def Solver(N,L,RE):
    N = N
    h = L/(N-1)
    
    
    Re = RE
    nu = 1/Re

    u_final = np.zeros([N,N])
    v_final = np.zeros([N,N])
    p_final = np.zeros([N,N])

    u_final[0,:] = 1
    u = np.zeros([N,N+1])
    u_star = np.zeros([N,N+1])
    d_e = np.zeros([N+1,N])
    v = np.zeros([N+1,N])
    v_star = np.zeros([N+1,N])
    d_n = np.zeros([N,N+1])
    p= np.zeros([N+1,N+1])
    p_star = np.zeros([N+1,N+1])
    p_star[N,N]=1
    pc = np.zeros([N+1,N+1])
    b= np.zeros([N+1,N+1])

    u[0,:]=2

    u_new = np.zeros([N,N+1])
    v_new = np.zeros([N+1,N])
    p_new = np.zeros([N+1,N+1])
    
    u_new[0,:]=2

    u,v,p=simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_n,d_e,N,nu,b,h)

    u_final = 0.5 * (u[:-1, :] + u[1:, :])
    v_final = 0.5 * (v[:, :-1] + v[:, 1:])
    p_final = 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])

    return u_final,v_final,p_final



def solve_pressure_correction(pc, b, a_E, a_W, a_N, a_S, a_P, alpha_p, p, d_e, d_n, h, max_iter=1000, tol=1e-6):
    """
    Jacobi solver for pressure correction equation with detailed debugging.
    """
    N, M = pc.shape
    pc_new = np.zeros_like(pc)  # Temporary array for updated values
    p_new=np.zeros_like(pc)

    for iter_jacobi in range(max_iter):
        max_residual = 0  # Track maximum residual for convergence
        for i in range(1, N-1):
            for j in range(1, M-1):
                # Check coefficients
                a_E[i, j] = -d_e[i,j]/h
                a_W[i, j] = -d_e[i,j-1]/h
                a_N[i, j] = -d_n[i-1,j]/h
                a_S[i, j] = -d_n[i,j]/h
                a_P[i, j] = a_E[i, j] + a_W[i, j] + a_N[i, j] + a_S[i, j]
                if a_P[i, j] <= 0:
                    a_P[i, j] = max(a_P[i, j], 1e-10)

                # Jacobi update formula
                pc_new[i, j] = (a_E[i, j] * pc[i, j+1] +
                                a_W[i, j] * pc[i, j-1] +
                                a_N[i, j] * pc[i-1, j] +
                                a_S[i, j] * pc[i+1, j] +
                                b[i, j]) / a_P[i, j]
                
                # Compute residual
                residual = abs(pc_new[i, j] - pc[i, j])
                max_residual = max(max_residual, residual)

        # Swap arrays for next iteration
        pc[:, :] = pc_new[:, :]

        # Convergence check
        if max_residual < tol:
            # correction of the pressure field
            # Update pressure field after convergence
            p_new[1:N-1, 1:M-1] = p[1:N-1, 1:M-1] + alpha_p * pc[1:N-1, 1:M-1]

            # Enforce boundary conditions on p_new
            p_new[0, :] = p_new[1, :]  # Top boundary
            p_new[-1, :] = p_new[-2, :]  # Bottom boundary
            p_new[:, 0] = p_new[:, 1]  # Left boundary
            p_new[:, -1] = p_new[:, -2]  # Right boundary
            print(f"Jacobi converged after {iter_jacobi} iterations")
            return p_new

    print(f"Jacobi solver did not converge after {max_iter} iterations. Final residual: {max_residual}")
    return pc
