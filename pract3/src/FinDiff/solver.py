import numpy as np
def central_difference_x(f, h):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * h)
    return diff

def central_difference_y(f, h):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * h)
    return diff

def laplace(f, h):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, :-2] + f[:-2, 1:-1] - 4 * f[1:-1, 1:-1] + f[1:-1, 2:] + f[2:, 1:-1]) / (h ** 2)
    return diff

def poisson(p, rhs, h, tol=1e-5, max_iter=10000):
    global iterations
    iterations = 0
    pn1 = p.copy()
    for _ in range(max_iter):
        p_old = pn1.copy()
        pn1[1:-1, 1:-1] = 0.25 * (p_old[1:-1, :-2] + p_old[:-2, 1:-1] + p_old[1:-1, 2:] + p_old[2:, 1:-1] - h**2 * rhs[1:-1, 1:-1])
        
        # Boundary conditions
        pn1[:, 0] = pn1[:, 1]  # dp/dy = 0 at y = 0
        pn1[:, -1] = pn1[:, -2]  # dp/dy = 0 at y = 1
        pn1[0, :] = pn1[1, :]  # dp/dx = 0 at x = 0
        pn1[-1, :] = pn1[-2, :]  # dp/dx = 0 at x = 1
        
        # Check for convergence
        error = np.linalg.norm(pn1 - p_old, ord=np.inf)
        if error < tol:
            break
        
        # # Error monitoring after every few timesteps
        # if iterations % 5000 == 0:
        #     import matplotlib.pyplot as plt
        #     plt.figure(1)
        #     plt.semilogy(iterations, error, '--or')
        #     plt.xlabel('Iterations')
        #     plt.ylabel('Residual Error')
            # plt.show()
        
        iterations += 1
        p = pn1.copy()
    return pn1