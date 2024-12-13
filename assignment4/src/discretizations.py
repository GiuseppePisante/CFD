import numpy as np
import matplotlib.pyplot as plt

def discretization1(N, dx, Pe):
    A = np.zeros((N, N))
    A += np.diag(-2 * np.ones(N))
    A[0,0] = 1
    A[-1, -1] = 1
    A += np.diag((1 - Pe * dx / 2) * np.ones(N - 1), 1)
    A[0, 1] = 0
    A += np.diag((1 + Pe * dx / 2) * np.ones(N - 1), -1)
    A[-1, -2] = 0
    b = np.zeros(N)
    b[-1] = 1
    return A, b

def discretization2(N, dx, Pe):
    A = np.zeros((N, N))
    A += np.diag(-(2 + dx * Pe) * np.ones(N))
    A[0,0] = 1
    A[-1, -1] = 1
    A += np.diag(np.ones(N - 1), 1)
    A[0, 1] = 0
    A += np.diag((1 + Pe * dx) * np.ones(N - 1), -1)
    A[-1, -2] = 0
    b = np.zeros(N)
    b[-1] = 1
    return A, b


def discretization3(N, dx, Pe):
    A = np.zeros((N, N))
    A += np.diag((-2 - 3 * Pe * dx / 2) * np.ones(N))
    A[0,0] = 1
    A[-1, -1] = 1
    A += np.diag(np.ones(N - 1), 1)
    A[0, 1] = 0
    A += np.diag(-Pe / 2 * np.ones(N - 2), -2)
    A[-1, -3] = 0
    A += np.diag( (1 + 2 * Pe * dx) * np.ones(N - 1), -1)
    A[-1, -2] = 0
    b = np.zeros(N)
    b[-1] = 1
    return A, b
