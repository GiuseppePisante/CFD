import numpy as np
from scipy.linalg import solve_banded

def solver_FDM(n, x, Pe):

    A = np.zeros((n, n))
    b = np.zeros(n)  

    for i in range(1, n-1):
        dx = x[i+1] - x[i-1]
        f = x[i+1] -2*x[i] +x[i-1]

        a_i = 1/dx - 4/(Pe*dx**2) + 4*f/(Pe*dx**3)
        c_i = -1/dx - 4/(Pe*dx**2) - 4*f/(Pe*dx**3)
        b_i = -(a_i + c_i)
        
        A[i, i-1] = c_i
        A[i, i] = b_i
        A[i, i+1] = a_i

    # Boundary conditions
    A[0, 0] = 1  
    A[-1, -1] = 1  
    b[0] = 0
    b[-1] = 1

    return A, b

def solver_FVM(N, Pe, u):
    x = np.linspace(0, 1, N) ** 0.7
    dx = np.diff(x) 
    
    xf = (x[:-1] + x[1:]) / 2
    w = dx[:-1] / (x[1:-1] - xf[:-1])
    w_prime = dx[1:] / (x[2:] - xf[1:])
    
    a_lower = -1 / (Pe * dx[:-1]) - u * w_prime  
    a_main = (1 / (Pe * dx[:-1]) + 1 / (Pe * dx[1:])) + u * w - u * (1 - w_prime)  
    a_upper = -1 / (Pe * dx[1:]) - u * (1 - w)  
    
    A_banded = np.zeros((3, N - 2))
    A_banded[0, 1:] = a_upper[:-1]  
    A_banded[1, :] = a_main  
    A_banded[2, :-1] = a_lower[1:] 
    b = np.zeros(N - 2)
    
    Phi_0, Phi_N = 0.0, 1.0 
    b[0] -= a_lower[0] * Phi_0
    b[-1] -= a_upper[-1] * Phi_N
    
    Phi_inner = solve_banded((1, 1), A_banded, b)
    
    Phi_num = np.hstack([Phi_0, Phi_inner, Phi_N])
    return x, Phi_num