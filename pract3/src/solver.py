import numpy as np

def simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_n,d_e,N,nu,b,h):

    # Under-relaxation factors
    alpha = 0.7
    alpha_p = 0.3
    err =1
    it=0
    itermax=1000
    tol = 1e-2

    while ((err > tol) & (it<itermax)):

        for i in range(1,N):
            for j in range(1,N-1):
                u_E = 0.5*(u[i,j] + u[i,j+1])
                u_W = 0.5*(u[i,j] + u[i,j-1])
                v_N = 0.5*(v[i-1,j] + v[i-1,j+1])
                v_S = 0.5*(v[i,j] + v[i,j+1])

                a_E = -0.5*u_E*h + nu
                a_W = 0.5*u_W*h + nu
                a_N = -0.5*v_N*h + nu
                a_S = 0.5*v_S*h + nu

                a_e = 0.5*u_E*h - 0.5*u_W*h + 0.5*v_N*h - 0.5*v_S*h + 4*nu

                A_e = -h
                d_e[i,j] = A_e/a_e

                u_star_mid = (a_E*u[i,j+1] + a_W*u[i,j-1] + a_N*u[i-1,j] + a_S*u[i+1,j])/a_e + d_e[i,j]*(p[i,j+1] - p[i,j])
                u_star[i,j] = (1-alpha)*u[i,j] + alpha*u_star_mid

        # Boundary x-momentum
        u_star[0,:] = 2 - u_star[1,:]
        u_star[N ,:] = -u_star[N-1,:]
        u_star[1:N-1,0] = 0
        u_star[1:N-1,N-1] = 0

        for i in range(1,N - 1):
            for j  in range(1,N):
                u_E = 0.5*(u[i,j] + u[i+1,j])
                u_W = 0.5*(u[i,j-1] + u[i+1,j-1])
                v_N = 0.5*(v[i-1,j] + v[i,j])
                v_S = 0.5*(v[i,j] + v[i+1,j])

                a_E = -0.5*u_E*h + nu
                a_W = 0.5*u_W*h + nu
                a_N = -0.5*v_N*h + nu
                a_S = 0.5*v_S*h + nu

                a_n = 0.5*u_E*h - 0.5*u_W*h + 0.5*v_N*h - 0.5*v_S*h + 4*nu

                A_n = -h
                d_n[i,j] = A_n/a_n

                v_star_mid = (a_E*v[i,j+1] + a_W*v[i,j-1] + a_N*v[i-1,j] + a_S*v[i+1,j])/a_n + d_n[i,j]*(p[i,j] - p[i+1,j])
                v_star[i,j] = (1-alpha)*v[i,j] + alpha*v_star_mid

        # Boundary y-momentum eq.
        v_star[:,0] = -v_star[:,1]
        v_star[:,N] = -v_star[:,N-1]
        v_star[0,1:N-1] = 0
        v_star[N-1,1:N-1] = 0

        # Correction term
        pc[0:N,0:N]=0
 
        # Interior pressure correction
        for i in range(1,N):
            for j in range(1,N):
                a_E = -d_e[i,j]*h
                a_W = -d_e[i,j-1]*h
                a_N = -d_n[i-1,j]*h
                a_S = -d_n[i,j]*h
                a_P = a_E + a_W + a_N + a_S
                b[i,j] = -(u_star[i,j] - u_star[i,j-1])*h + (v_star[i,j] - v_star[i-1,j])*h

                pc[i,j] = (a_E*pc[i,j+1] + a_W*pc[i,j-1] + a_N*pc[i-1,j] + a_S*pc[i+1,j] + b[i,j])/a_P

        # Pressure field Correction
        p_new[1:N, 1:N] = p[1:N, 1:N] + alpha_p * pc[1:N, 1:N]

        # Boundary Continuity
        p_new[0,:] = p_new[1,:]
        p_new[N,:] = p_new[N-1,:]
        p_new[:,0] = p_new[:,1]
        p_new[:,N]= p_new[:,N-1]

        # Velocity Correction
        u_new[1:N, 1:N-1] = u_star[1:N, 1:N-1] + d_e[1:N, 1:N-1] * (pc[1:N, 2:N] - pc[1:N, 1:N-1])


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
        err = np.sum(np.abs(b[1:N, 1:N]))
        u = u_new.copy()
        v = v_new.copy()
        p = p_new.copy()
        it = it + 1

    return u,v,p



def Solver(N,L,RE):
    N = N
    dom_length = L
    h = dom_length/(N-1)
    
    Re = RE
    nu = 1/Re

    u_final = np.zeros([N,N])
    v_final = np.zeros([N,N])
    p_final = np.zeros([N,N])

    u_final[0,:] = 1
    u = np.zeros([N+1,N])
    u_star = np.zeros([N+1,N])
    d_e = np.zeros([N+1,N])
    v = np.zeros([N,N+1])
    v_star = np.zeros([N,N+1])
    d_n = np.zeros([N,N+1])
    p= np.zeros([N+1,N+1])
    p_star = np.zeros([N+1,N+1])
    p_star[N,N]=1
    pc = np.zeros([N+1,N+1])
    b= np.zeros([N+1,N+1])

    u[0,:]=2

    u_new = np.zeros([N+1,N])
    v_new = np.zeros([N,N+1])
    p_new = np.zeros([N+1,N+1])
    
    u_new[0,:]=2

    u,v,p=simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_n,d_e,N,nu,b,h)

    u_final = 0.5 * (u[:-1, :] + u[1:, :])
    v_final = 0.5 * (v[:, :-1] + v[:, 1:])
    p_final = 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])

    return u_final,v_final,p_final