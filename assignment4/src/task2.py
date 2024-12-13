import numpy as np
import matplotlib.pyplot as plt
import discretizations as disc

# Task 4.2

### (a)
def analytical_solution(x, Pe):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

m = np.linspace(2, 7, 6)
dx_vec = 1 / 2**m
Pe = 25
N_vec = (1 / dx_vec).astype(int) + 1
res1 = []
res2 = []
res3 = []
L2norm_1 = []
L2norm_2 = []
L2norm_3 = []

for i in range(len(N_vec)):
    N = N_vec[i]
    dx = dx_vec[i]
    x = np.linspace(0, 1, N)
    u = analytical_solution(x, Pe)

    A, b = disc.discretization1(N, dx, Pe)
    res = np.linalg.solve(A, b)
    res1.append(res)
    L2norm_1.append(np.linalg.norm(res - u))

    A, b = disc.discretization2(N, dx, Pe)
    res = np.linalg.solve(A, b)
    res2.append(res)
    L2norm_2.append(np.linalg.norm(res - u))

    A, b = disc.discretization3(N, dx, Pe)
    res = np.linalg.solve(A, b)
    res3.append(res)
    L2norm_3.append(np.linalg.norm(res - u))


fig, axs = plt.subplots(len(N_vec)//3, 3, figsize=(15, 15))
for i in range(len(N_vec)):
    ax = axs[i//3, i%3]
    ax.plot(res1[i], label='discretization1')
    ax.plot(res2[i], label='discretization2')
    ax.plot(res3[i], label='discretization3')
    ax.set_title(f'N={N_vec[i]}')
    ax.legend()
plt.savefig('task2.png')

plt.figure()
plt.loglog(dx_vec, L2norm_1, label='discretization1')
plt.loglog(dx_vec, L2norm_2, label='discretization2')
plt.loglog(dx_vec, L2norm_3, label='discretization3')
plt.xlabel('N')
plt.ylabel('L2 norm')
plt.legend()
plt.savefig('task2_L2norm.png')



### (b)
Pe_values = [25, 50, 100, 120, 125, 128, 130, 135, 150, 200, 400]
dx = 1/64
N = int(1/dx) + 1
x = np.linspace(0, 1, N)

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, Pe in enumerate([25, 50, 100]):
    u = analytical_solution(x, Pe)
    
    A, b = disc.discretization1(N, dx, Pe)
    res1 = np.linalg.solve(A, b)
    
    A, b = disc.discretization2(N, dx, Pe)
    res2 = np.linalg.solve(A, b)
    
    A, b = disc.discretization3(N, dx, Pe)
    res3 = np.linalg.solve(A, b)
    
    axs[0, i].plot(x, res1, label='discretization1')
    axs[0, i].plot(x, res2, label='discretization2')
    axs[0, i].plot(x, res3, label='discretization3')
    axs[0, i].set_title(f'Pe={Pe}')
    axs[0, i].legend()

for i, Pe in enumerate([120, 125, 128, 130, 135]):
    u = analytical_solution(x, Pe)
    
    A, b = disc.discretization1(N, dx, Pe)
    res1 = np.linalg.solve(A, b)
    
    A, b = disc.discretization2(N, dx, Pe)
    res2 = np.linalg.solve(A, b)
    
    A, b = disc.discretization3(N, dx, Pe)
    res3 = np.linalg.solve(A, b)
    
    axs[1, i % 3].plot(x, res1, label='discretization1')
    axs[1, i % 3].plot(x, res2, label='discretization2')
    axs[1, i % 3].plot(x, res3, label='discretization3')
    axs[1, i % 3].set_title(f'Pe={Pe}')
    axs[1, i % 3].legend()

for i, Pe in enumerate([150, 200, 400]):
    u = analytical_solution(x, Pe)
    
    A, b = disc.discretization1(N, dx, Pe)
    res1 = np.linalg.solve(A, b)
    
    A, b = disc.discretization2(N, dx, Pe)
    res2 = np.linalg.solve(A, b)
    
    A, b = disc.discretization3(N, dx, Pe)
    res3 = np.linalg.solve(A, b)
    
    axs[2, i].plot(x, res1, label='discretization1')
    axs[2, i].plot(x, res2, label='discretization2')
    axs[2, i].plot(x, res3, label='discretization3')
    axs[2, i].set_title(f'Pe={Pe}')
    axs[2, i].legend()

plt.tight_layout()
plt.savefig('task2_fixed_dx.png')
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, Pe in enumerate([25, 50, 100]):
    u = analytical_solution(x, Pe)
    
    A, b = disc.discretization1(N, dx, Pe)
    res1 = np.linalg.solve(A, b)
    axs[0, 0].plot(x, res1, label=f'Pe={Pe}')
    axs[0, 0].set_title('discretization1')
    axs[0, 0].legend()

    A, b = disc.discretization2(N, dx, Pe)
    res2 = np.linalg.solve(A, b)
    axs[1, 0].plot(x, res2, label=f'Pe={Pe}')
    axs[1, 0].set_title('discretization2')
    axs[1, 0].legend()

    A, b = disc.discretization3(N, dx, Pe)
    res3 = np.linalg.solve(A, b)
    axs[2, 0].plot(x, res3, label=f'Pe={Pe}')
    axs[2, 0].set_title('discretization3')
    axs[2, 0].legend()

for i, Pe in enumerate([120, 125, 128, 130, 135]):
    u = analytical_solution(x, Pe)
    
    A, b = disc.discretization1(N, dx, Pe)
    res1 = np.linalg.solve(A, b)
    axs[0, 1].plot(x, res1, label=f'Pe={Pe}')
    axs[0, 1].set_title('discretization1')
    axs[0, 1].legend()

    A, b = disc.discretization2(N, dx, Pe)
    res2 = np.linalg.solve(A, b)
    axs[1, 1].plot(x, res2, label=f'Pe={Pe}')
    axs[1, 1].set_title('discretization2')
    axs[1, 1].legend()

    A, b = disc.discretization3(N, dx, Pe)
    res3 = np.linalg.solve(A, b)
    axs[2, 1].plot(x, res3, label=f'Pe={Pe}')
    axs[2, 1].set_title('discretization3')
    axs[2, 1].legend()

for i, Pe in enumerate([150, 200, 400]):
    u = analytical_solution(x, Pe)
    
    A, b = disc.discretization1(N, dx, Pe)
    res1 = np.linalg.solve(A, b)
    axs[0, 2].plot(x, res1, label=f'Pe={Pe}')
    axs[0, 2].set_title('discretization1')
    axs[0, 2].legend()

    A, b = disc.discretization2(N, dx, Pe)
    res2 = np.linalg.solve(A, b)
    axs[1, 2].plot(x, res2, label=f'Pe={Pe}')
    axs[1, 2].set_title('discretization2')
    axs[1, 2].legend()

    A, b = disc.discretization3(N, dx, Pe)
    res3 = np.linalg.solve(A, b)
    axs[2, 2].plot(x, res3, label=f'Pe={Pe}')
    axs[2, 2].set_title('discretization3')
    axs[2, 2].legend()

plt.tight_layout()
plt.savefig('task2_fixed_dx.png')


