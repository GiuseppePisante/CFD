import numpy as np

def simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_y,d_x,N,nu,b,h):

    # Under-relaxation factors
    alpha = 0.01
    alpha_p = 0.8
    err_u, err_v, err_p =1, 1, 1
    it=0
    itermax=10
    tol = 1e-2

    while ((it<itermax)):
        for i in range(1, N-1):  # Loop over internal cells
            for j in range(1, N):
                # Interpolated face velocities
                u_E = 0.5*(u[i,j] + u[i,j+1])
                u_W = 0.5*(u[i,j] + u[i,j-1])
                v_E = 0.5*(v[i+1,j] + v[i,j])
                v_W = 0.5*(v[i+1,j-1] + v[i,j-1])

                # Coefficients (Convective and Diffusive)
                a_C = 0.5 * u_E * h - 0.5 * u_W * h + 0.5 * v_E * h - 0.5 * v_W * h + 2* nu
                a_N= - 0.5 * nu
                a_S = - 0.5 * nu
                a_E = 0.5 * u_E * h + 0.5 * v_E * h - 0.5 * nu 
                a_W = - 0.5 * u_W * h - 0.5 * v_W * h - 0.5 * nu

                d_x[i,j] = -h/a_C

                # Momentum equation (Intermediate u*)
                u_star[i,j] = (1-alpha) * u[i,j] + alpha*(a_E * u[i,j+1] + a_W * u[i,j-1] + a_S * u[i-1,j] + a_N * u[i+1,j] + h * (p[i,j] - p[i,j-1])) / a_C
        # Boundary conditions
        u_star[0, :] = u_star[1,:]/3
        u_star[N - 1, :] = 2/3 + u_star[N - 2,:]/3
        u_star[1 : N - 2, 0] = 0
        u_star[1 : N - 2, N - 1] = 0

        for i in range(1, N):  # Loop over internal cells
            for j in range(1, N-1):
                # Interpolated face velocities
                u_N = 0.5*(u[i,j+1] + u[i,j])
                u_S = 0.5*(u[i-1,j+1] + u[i-1,j])
                v_N = 0.5*(v[i,j+1] + v[i,j])
                v_S = 0.5*(v[i,j-1] + v[i,j])

                # Coefficients (Convective and Diffusive)
                a_C = 0.5 * v_S * h - 0.5 * v_N * h + 0.5 * u_N * h - 0.5 * u_S * h + 2* nu
                a_N = 0.5 * v_N * h + 0.5 * u_N * h - 0.5 * nu
                a_S = - 0.5 * v_S * h - 0.5 * u_S * h - 0.5 * nu
                a_E = - 0.5 * nu
                a_W = - 0.5 * nu
                
                d_y[i,j] = -h/a_C

                # Momentum equation (Intermediate u*)
                v_star[i,j]= (1-alpha) * v[i,j] + alpha*(a_E * v[i,j+1] + a_W * v[i,j-1] + a_S * v[i-1,j] + a_N * v[i+1,j] + h * (p[i,j] - p[i-1,j])) / a_C

        # Boundary Conditions for v
        v_star[1 : N, 0] = v_star[1 : N, 1]/3  # Left wall
        v_star[1 : N, N - 1] = v_star[1 : N, N - 2]/3  # Right wall
        v_star[0, :] = 0  # Top wall
        v_star[N, :] = 0  # Bottom wall

        # Correction term (Initialization)
        pc[0:N-1, 0:N-1] = 0
        
        #Continuity equation a.k.a. pressure correction - Interior
        for i in range(1,N-1):
            for j in range(1,N-1):
                a_E = -d_x[i,j]*h
                a_W = -d_x[i,j-1]*h
                a_N = -d_y[i-1,j]*h
                a_S = -d_y[i,j]*h
                a_P = a_E + a_W + a_N + a_S
                b[i,j] = -(u_star[i,j] - u_star[i,j-1])/h + (v_star[i,j] - v_star[i-1,j])/h

                pc[i,j] = (a_E*pc[i,j+1] + a_W*pc[i,j-1] + a_N*pc[i-1,j] + a_S*pc[i+1,j] + b[i,j])/a_P


        #Correcting the pressure field
        for i in range( 1,N):
            for j in range( 1,N) :
                p_new[i,j] = p[i,j] + alpha_p*pc[i,j]

        # Boundary Continuity
        p_new[0,:] = p_new[1,:]
        p_new[N-1,:] = p_new[N-2,:]
        p_new[:,0] = p_new[:,1]
        p_new[:,N-1]= p_new[:,N-2]

        # Velocity Correction
        for i in range(1, N - 1):
            for j in range(1, N):
                u_new[i,j] = u_star[i,j] + d_x[i,j] * (pc[i,j] - pc[i,j-1])

        # Boundary x-correction
        u_new[0,:] = u_new[1,:]/3
        u_new[N-1,:] = 2/3-u_new[N-2,:]/3
        u_new[1:N-2,0] = 0
        u_new[1:N-2,N-1] = 0

        for i in range(1, N):
            for j in range(1 , N - 1):
                v_new[i,j] = v_star[i,j] + d_y[i,j]*(pc[i,j] - pc[i-1,j])

        # Boundary y-momentum
        v_new[1 : N, 0] = v_new[1 : N, 1]/3  # Left wall
        v_new[1 : N, N - 1] = v_new[1 : N, N - 2]/3  # Right wall
        v_new[0,:] = 0
        v_new[N,:] = 0

        print(pc)
        
        # Residual computation
        err_u = np.abs(u_new - u).max() 
        err_v = np.abs(v_new - v).max()
        err_p = np.abs(p_new - p).max()
        print(f"Error u: {err_u}, Error v: {err_v}, Error p: {err_p}")
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
    d_x = np.zeros([N,N+1])
    v = np.zeros([N+1,N])
    v_star = np.zeros([N+1,N])
    d_y = np.zeros([N+1,N])
    p= np.zeros([N,N])
    p_star = np.zeros([N,N])
    p_star[N-1,N-1]=1
    pc = np.zeros([N,N])
    b= np.zeros([N,N])

    u[N - 1,:]=2/3

    u_new = np.zeros([N,N+1])
    v_new = np.zeros([N+1,N])
    p_new = np.zeros([N,N])

    u_new[N - 1,:]=2
    
    u,v,p=simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_y,d_x,N,nu,b,h)

    u_final = 0.5 * (u[:-1, :] + u[1:, :])
    v_final = 0.5 * (v[:, :-1] + v[:, 1:])
    p_final = 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])

    return u_final,v_final,p_final



def solve_pressure_correction(pc, b, a_E, a_W, a_N, a_S, a_P, alpha_p, p, d_x, d_y, h, max_iter=1000, tol=1e-6):
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
                a_E[i, j] = -d_x[i,j]/h
                a_W[i, j] = -d_x[i,j-1]/h
                a_N[i, j] = -d_y[i-1,j]/h
                a_S[i, j] = -d_y[i,j]/h
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
