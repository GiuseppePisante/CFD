import numpy as np

def Solver(N,L,RE):
    h = L/(N-1)
    Re = RE
    nu = 1/Re
    
    u=np.zeros(N*(N+1))
    v=np.zeros((N+1)*N)
    p=np.zeros(N*N)

    u_star=np.zeros(N*(N+1))
    v_star=np.zeros((N+1)*N)

    u_new=np.zeros(N*(N+1))
    v_new=np.zeros((N+1)*N)
    p_new=np.zeros(N*N)

    pc=np.zeros(N*N)

    b=np.zeros(N*(N+1))

    d_y=np.zeros((N+1, N))
    d_x=np.zeros((N, N+1))

    
    
    u,v,p=simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_y,d_x,N,nu,b,h)

    u_final = np.zeros([N,N])
    v_final = np.zeros([N,N])

    for i in range(N):
        for j in range(N):
            idv=i*N+j
            idu=i*(N+1)+j
            u_final[i,j] = 0.5*(u[idu] + u[idu + 1])
            v_final[i,j] = 0.5*(v[idv] + v[idv + N])
            
    p = p.reshape(N, N)
    return u_final,v_final,p

def simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_y,d_x,N,nu,b,h):
    A=np.zeros(((N+1)*N,(N+1)*N))
    alpha = 0.7
    alpha_p = 0.3

    it=0
    itermax=1
    tol = 1e-2

    while ((it<itermax)):
        for i in range(0, N):  # Loop over internal cells
            for j in range(0, N+1):
                idu= i * (N+1) + j
                idv= i * N + j

                if i == N - 1:
                    print(j)
                    A[idu,idu] = 1
                    A[idu,idu - N - 1] = - 1/3
                    b[idu] = 2/3
                elif i == 0:
                    A[idu,idu] = 1
                    A[idu,idu + N + 1] = - 1/3
                elif j == 0:
                    A[idu,idu] = 1
                elif j == N:
                    A[idu,idu] = 1
                else:
                    # Interpolated face velocities
                    u_E = 0.5*(u[idu] + u[idu+1])
                    u_W = 0.5*(u[idu] + u[idu-1])
                    v_E = 0.5*(v[idv + N] + v[idv])
                    v_W = 0.5*(v[idv + N - 1] + v[idv - 1])

                    # Coefficients (Convective and Diffusive)
                    A[idu ,idu + N + 1] = - 0.5 * nu    #a_N
                    A[idu,idu - N - 1] = - 0.5 * nu    #a_S
                    A[idu,idu + 1] = 0.5 * u_E * h + 0.5 * v_E * h - 0.5 * nu      #a_E
                    A[idu,idu - 1] = - 0.5 * u_W * h - 0.5 * v_W * h - 0.5 * nu    #a_W

                    A[idu, idu] = 0.5 * u_E * h - 0.5 * u_W * h + 0.5 * v_E * h - 0.5 * v_W * h + 2* nu  #a_C
                    b[idu] = h * (p[idu] - p[idu - 1])

                    d_x[i,j] = -h/A[idu, idu]
       
        u_star = jacobi(A,b,tol,u_star)

        for i in range(0, N):
            for j in range(0, N + 1):
                idu= i * (N+1)+ j
                u_star[idu] = u[idu] + alpha * u_star[idu]

        b=np.zeros(N*(N+1))
        A=np.zeros(((N+1)*N,(N+1)*N))

        # y-momentum
        for i in range(0, N + 1):  # Loop over internal cells
            for j in range(0, N):
                idu= i * (N+1) + j
                idv= i * N + j

                if j == N - 1:
                    A[idv, idv] = 1
                    A[idv,idv - 1] = - 1/3
                elif j == 0:
                    A[idv,idv] = 1
                    A[idv,idv + 1] = - 1/3
                elif i == 0:
                    A[idv,idv] = 1
                elif i == N:
                    A[idv,idv] = 1
                else:
                    # Interpolated face velocities
                    u_N = 0.5*(u[idu + 1] + u[idu])       #u_N
                    u_S = 0.5*(u[idu - N] + u[idu - N - 1])   #u_S
                    v_N = 0.5*(v[idv + 1] + v[idv])       #v_N
                    v_S = 0.5*(v[idv - 1] + v[idv])       #v_S

                    # Coefficients (Convective and Diffusive)
                    A[idv,idv + N] = 0.5 * v_N * h + 0.5 * u_N * h - 0.5 * nu   #a_N
                    A[idv,idv - N] = - 0.5 * v_S * h - 0.5 * u_S * h - 0.5 * nu        #a_S
                    A[idv,idv + 1] = - 0.5 * nu                                        #a_E
                    A[idv,idv - 1] = - 0.5 * nu                                        #a_W

                    A[idv,idv] = 0.5 * v_S * h - 0.5 * v_N * h + 0.5 * u_N * h - 0.5 * u_S * h + 2* nu
                    b[idv] = h * (p[idv] - p[idv - N])

                    d_y[i,j] = -h/A[idv,idv]


        
        v_star = jacobi(A,b,tol,v_star)
        

                
        for i in range(0, N + 1):
            for j in range(0, N ):
                idv= i * N +  j
                v_star[idv] = v[idv] + alpha * v_star[idv]



        # Pressure correction
        bp=np.zeros(N*N)
        P=np.zeros((N*N,N*N))
        for i in range(0, N):
            for j in range(0, N):
                idp = i * N + j
                idu = i * (N+1) + j

                if i == N - 1:
                    P[idp, idp] = 1
                    P[idp, idp - N]= -1
                elif i == 0:
                    P[idp, idp] = 1
                    P[idp, idp + N]= -1
                elif j == 0:
                    P[idp, idp] = 1
                    P[idp, idp + 1]= -1
                elif j == N - 1:
                    P[idp, idp] = 1
                    P[idp, idp - 1] = -1
                else:
                    a_E = -d_x[i, j+1] * h
                    a_W = -d_x[i, j] * h
                    a_N = -d_y[i+1, j] * h
                    a_S = -d_y[i, j] * h
                    a_P = a_E + a_W + a_N + a_S
                    P[idp, idp + 1] = a_E
                    P[idp, idp - 1] = a_W
                    P[idp, idp + N] = a_N
                    P[idp, idp - N] = a_S
                    P[idp, idp] = a_P
                    bp[idp] = a_E*pc[idp+1] + a_W*pc[idp-1] + a_N*pc[idp-N] + a_S*pc[idp+N] + \
                            -(u_star[idu + 1] - u_star[idu]) / h - (v_star[idp + N] - v_star[idp]) / h

        pc = jacobi(P, bp, tol, pc)

        for i in range(1, N-1):
            for j in range(1, N-1):
                idp = i * N +  j
                p_new[idp] = p[idp] + alpha_p * pc[idp]

        # Velocity Correction
        for i in range(1, N - 1):
            for j in range(1, N):
                idu = i * (N+1) + j
                idp = i * N +  j
                u_new[idu] = u_star[idu] + d_x[i,j] * (pc[idp] - pc[idp - 1])

        for i in range(1, N):
                for j in range(1 , N - 1):
                    idv= i * N + j
                    v_new[idv] = v_star[idv] + d_y[i,j]*(pc[idv] - pc[idv - N])

        
        # Residual computation
        err_u = np.abs(u_new - u).max() 
        err_v = np.abs(v_new - v).max()
        err_p = np.abs(p_new - p).max()

        u = u_new.copy()
        v = v_new.copy()
        p = p_new.copy()
        it = it + 1

    return u,v,p


def jacobi(A, b, tol=1e-6, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(b))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Iterate for N times
    err = 1
    it=0

    while err > tol:
        x_new = (b - np.dot(R, x)) / D
        # print(max(x_new))
        err = np.linalg.norm(x_new - x)
        x = x_new
        it=it+1

    print(f"Jacobi iterations: {it}")
    return x
