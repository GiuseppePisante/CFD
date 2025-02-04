import numpy as np

def discretization(n, x, Pe):

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