import numpy as np

def lid_driven_solver(N,L,RE):

    h = L/(N-1)
    nu = 1/RE

    # Under-relaxation factors
    alpha = 0.01
    alpha_p = 0.001

    u_final = np.zeros([N,N])
    v_final = np.zeros([N,N])
    p_final = np.zeros([N,N])

    u=np.zeros(N*(N+1))
    v=np.zeros((N+1)*N)
    p=np.zeros((N+1)*(N+1))

    u_star=np.zeros(N*(N+1))
    v_star=np.zeros((N+1)*N)

    u_new=np.zeros(N*(N+1))
    v_new=np.zeros((N+1)*N)
    p_new=np.zeros((N+1)*(N+1))

    b=np.zeros(N*(N+1))

    d_y=np.zeros((N+1, N))
    d_x=np.zeros((N, N+1))

    p_star = np.zeros((N+1)*(N+1))
    p_star[N*N+2*N]=1
    pc = np.zeros((N+1)*(N+1))

    u[0:N]=2
    u_new[0:N]=2

    u,v,p=simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_y,d_x,N,alpha_p,alpha,nu,b,h)

    for i in range(N):
        for j in range(N):
            idu=i*N+j
            idv=i*(N+1)+j
            u_final[i,j] = 0.5*(u[idu] + u[idu + N])
            v_final[i,j] = 0.5*(v[idv] + v[idv + 1])
            p_final[i,j] = 0.25*(p[idv] + p[idv+1] + p[idv + N + 1] + p[idv + N + 2])

    return u_final,v_final,p_final



def simple_solver(u,v,p,u_star,v_star,u_new,v_new,p_new,pc,d_y,d_x,N,alpha_p,alpha,nu,b,h):

    it=0
    itermax=10

    tol = 1e-6
    err_u = 1
    err_v = 1
    err_p = 1

    while it<itermax and (err_u > tol or err_v > tol or err_p > tol):
        
        A=np.zeros([N*(N+1),N*(N+1)])
        b=np.zeros(N*(N+1))
        for i in range(0, N + 1):
            for j in range(0, N):
                idu= i * N + j
                idv= i * (N + 1) + j

                if i == 0:
                    A[idu,idu] = 1
                    A[idu,idu + N] = 1
                    b[idu] = 2
                elif i == N:
                    A[idu,idu] = 1
                    A[idu,idu - N] = 1
                elif j == 0:
                    A[idu,idu] = 1
                elif j == N - 1:
                    A[idu,idu] = 1
                else:
                    u_E = 0.5*(u[idu] + u[idu + 1])
                    u_W = 0.5*(u[idu] + u[idu - 1])
                    v_N = 0.5*(v[idv - N - 1] + v[idv - N])
                    v_S = 0.5*(v[idv] + v[idv + 1])

                    A[idu,idu + 1] =  0.5 * u_E * h - nu
                    A[idu,idu - 1] = -0.5 * u_W * h - nu
                    A[idu,idu - N] =  0.5 * v_N * h - nu
                    A[idu ,idu + N] = -0.5 * v_S * h - nu

                    A[idu, idu] = 0.5*u_E*h - 0.5*u_W*h + 0.5*v_N*h - 0.5*v_S*h + 4*nu

                    b[idu] = - h * (p[idu + 1] - p[idu])

                    d_x[i,j] = - h / A[idu, idu]                

        u_star = jacobi(A,b,tol,u_star)

        for i in range( 1,N ):
            for j in range(1,N - 1):
                idu= i * N + j
                u_star[idu] = (1-alpha)*u[idu] + alpha*u_star[idu]

        # x-momentum boundaries
        for i in range(0, N):
            u_star[i * N] = 0
            u_star[i * N + N] = 0

        u_star[0:N] = 2 - u_star[N:N + N]
        u_star[N * N:N * N + N + 1] = - u_star[N * N - N:N * N]
        
        # Y-Momentum
        A=np.zeros([N*(N+1),N*(N+1)])
        b=np.zeros(N*(N+1))
        for i in range(0,N):
            for j  in range(0 , N + 1):
                idu= i * N + j
                idv= i * (N+1) + j
                if j == N:
                    A[idv, idv] = 1
                    A[idv, idv - 1] = 1

                elif j == 0:
                    A[idv, idv] = 1
                    A[idv, idv + 1] = 1

                elif i == 0:
                    A[idv, idv] = 1
                elif i == N - 1:
                    A[idv, idv] = 1
                else:
                    u_E = 0.5*(u[idu] + u[idu + N])
                    u_W = 0.5*(u[idu - 1] + u[idu + N - 1])
                    v_N = 0.5*(v[idv - N - 1] + v[idv])
                    v_S = 0.5*(v[idv] + v[idv + N + 1])

                    A[idv,idv + 1] = 0.5 * u_E * h - nu
                    A[idv,idv - 1] = -0.5 * u_W * h - nu
                    A[idv,idv - N - 1] = 0.5 * v_N * h - nu
                    A[idv,idv + N + 1] = - 0.5 * v_S * h - nu

                    A[idv,idv] = 0.5 * u_E * h - 0.5 * u_W * h + 0.5 * v_N * h - 0.5 * v_S * h + 4 * nu
                    b[idv] = - h * (p[idv] - p[idv + N + 1])
                    
                    d_y[i,j] = -h / A[idv,idv]

        v_star = jacobi(A,b,tol,v_star)

        for i in range( 1,N - 1):
            for j in range(1,N):
                idv= i * (N+1) + j
                v_star[idv] = (1-alpha)*v[idv] + alpha*v_star[idv]

        # y-momentum boundaries
        for i in range(0, N):
            v_star[i * (N+1)] = 0
            v_star[i * (N+1) + N] = 0

        v_star[0:N+1] = 0
        v_star[N*(N+1):N*(N+1)+N] = 0
    


        # Zeroing the corrections to begin with
        pc=np.zeros((N+1)*(N+1))
        tmp=np.zeros((N-1)*(N-1))
        bp=np.zeros((N-1)*(N-1))
        P=np.zeros(((N-1)*(N-1),(N-1)*(N-1)))

        #Continuity equation a.k.a. pressure correction - Interior
        for i in range(1, N):
            for j in range(1, N):
                idu = i * N + j
                idp = (i - 1) * (N - 1) + j - 1
                idv = i * (N + 1) + j
                
                a_E = d_x[i,j] * h
                a_W = d_x[i,j-1] * h
                a_N = d_y[i-1,j] * h
                a_S = d_y[i,j]*h

                if i == 1 and j == 1:
                    a_N = 0
                    a_W = 0
                    P[idp, idp + 1] = a_E
                    P[idp, idp + N - 1] = a_S
                elif i == 1 and j == N - 1:
                    a_N = 0
                    a_E = 0
                    P[idp, idp - 1] = a_W
                    P[idp, idp + N - 1] = a_S
                elif i == N - 1 and j == 1:
                    a_S = 0
                    a_W = 0
                    P[idp, idp + 1] = a_E
                    P[idp, idp - N + 1] = a_N
                elif i == N - 1 and j == N - 1:
                    a_S = 0
                    a_E = 0
                    P[idp, idp - 1] = a_W
                    P[idp, idp - N + 1] = a_N
                elif i == 1:
                    a_N = 0
                elif i == N - 1:
                    a_S = 0
                    P[idp, idp + 1] = a_E
                    P[idp, idp - 1] = a_W
                    P[idp, idp - N + 1] = a_N
                elif j == 1:
                    a_W = 0
                    P[idp, idp + 1] = a_E
                    P[idp, idp - N + 1] = a_N
                    P[idp, idp + N - 1] = a_S
                elif j == N - 1:
                    a_E = 0
                    P[idp, idp - 1] = a_W
                    P[idp, idp - N + 1] = a_N
                    P[idp, idp + N - 1] = a_S
                else:
                    P[idp, idp + 1] = a_E
                    P[idp, idp - 1] = a_W
                    P[idp, idp - N + 1] = a_N
                    P[idp, idp + N - 1] = a_S

                P[idp, idp] = -(a_E + a_W + a_N + a_S) 
                bp[idp] = -(u_star[idu] - u_star[idu-1]) * h + (v_star[idv] - v_star[idv - N - 1]) * h

        tmp = jacobi(P, bp, tol, tmp)
        #Correcting the pressure field
        for i in range( 1,N):
            for j in range( 1,N) :
                idp = i * (N+1) + j
                idt = (i - 1) * (N - 1) + j - 1
                p_new[idp] = p[idp] + alpha_p*tmp[idt]


        for i in range(0, N + 1):
            p_new[i * (N+1)] = p_new[i * (N+1) + 1]
            p_new[i * (N+1) + N] = p_new[i * (N+1) + N - 1]
        p_new[0:N+1] = p_new[N+1:2*N+2]
        p_new[N*(N+1):N*(N+1)+N] = p_new[N*(N+1)-N - 1:N*(N+1)-1]
        
        # Correcting the velocities
        for i in range(1, N):
            for j in range( 1 , N - 1):
                idu = i * N + j
                idp = i * (N+1) + j
                idt = (i - 1) * (N - 1) + j - 1
                u_new[idu] = u_star[idu] + d_x[i,j]*(tmp[idt+1] - tmp[idt])

        for i in range(0, N):
            u_new[i * N] = 0
            u_new[i * N + N] = 0
        u_new[0:N] = 2 - u_new[N:N + N]
        u_new[N * N:N * N + N + 1] = - u_new[N * N - N:N * N]

        for i in range(1,N - 1):
            for j in range(1,N):
                idp = i * (N+1) + j
                idt = (i - 1) * (N - 1) + j - 1
                v_new[idp] = v_star[idp] + d_y[i,j]*(tmp[idt] - tmp[idt + N - 1])

    
        for i in range(0, N):
            v_new[i * (N+1)] = - v_new[i * (N+1)+ 1] 
            v_new[i * (N+1) + N] = - v_new[i * (N+1) + N - 1]
        v_new[0:N+1] = 0
        v_new[N*(N+1):N*(N+1)+N] = 0

        
        # Continuity residual as error measure
        # Residual computation
        err_u = np.abs(u_new - u).max() 
        err_v = np.abs(v_new - v).max()
        err_p = np.abs(p_new - p).max()

        u = u_new.copy()
        v = v_new.copy()
        p = p_new.copy()
        it = it + 1

    print(f"Converged in {it} iterations with error {err_u:.2e} {err_v:.2e} {err_p:.2e}")
    return u,v,p


import numpy as np

def jacobi(A, b, tol=1e-6, x=None, itermax=2000):
    """Risoluzione del sistema lineare Ax = b con il metodo di Jacobi."""
    
    # Controllo che A sia quadrata e compatibile con b
    n = len(A)
    if A.shape[0] != A.shape[1] or len(b) != n:
        raise ValueError("La matrice A deve essere quadrata e compatibile con il vettore b.")

    # Controllo che la diagonale non contenga zeri
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("La matrice A ha elementi nulli sulla diagonale, impossibile applicare Jacobi.")

    # Controllo della convergenza (criterio di dominanza diagonale)
    if not np.all(2 * np.abs(D) >= np.sum(np.abs(A), axis=1)):
        print("Avviso: la matrice A potrebbe non soddisfare il criterio di dominanza diagonale. Il metodo potrebbe non convergere.")

    # Inizializzazione della soluzione
    if x is None:
        x = np.zeros_like(b, dtype=np.float64)
    
    R = A - np.diagflat(D)
    
    err = float('inf')
    it = 0

    while err > tol and it < itermax:
        x_new = (b - np.dot(R, x)) / D
        err = np.linalg.norm(x_new - x, np.inf)  # Norma infinito per il criterio di arresto
        x = x_new
        it += 1

    print(f"Jacobi convergenza in {it} iterazioni con errore {err:.2e}")
    return x

