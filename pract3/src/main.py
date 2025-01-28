from solver import Solver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

L=1
N=100
Re=1000

L=1
N=10
RE=1

u,v,p=Solver(N,L,Re)

h = L/(N-1)
x = np.linspace(0,L,N)
y = np.linspace(0,L,N)
y_dom = 1 - ((np.arange(1, N + 1) - 1) * h)

X_u, Y_u = np.meshgrid(np.linspace(0, L, N+1), np.linspace(0, L, N-1))
X_v, Y_v = np.meshgrid(np.linspace(0, L, N-1), np.linspace(0, L, N+1))
X_p, Y_p = np.meshgrid(np.linspace(0, L, N-1), np.linspace(0, L, N-1))

vorticity_list=[]


fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot u
c1 = ax[0].contourf(X_u, Y_u, u, cmap=cm.viridis)
fig.colorbar(c1, ax=ax[0])
ax[0].set_title('Velocity u')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

# Plot v
c2 = ax[1].contourf(X_v, Y_v, v, cmap=cm.viridis)
fig.colorbar(c2, ax=ax[1])
ax[1].set_title('Velocity v')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

# Plot p
c3 = ax[2].contourf(X_p, Y_p, p, cmap=cm.viridis)
fig.colorbar(c3, ax=ax[2])
ax[2].set_title('Pressure p')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')

# Plot streamlines
# Interpolate u and v onto a common grid (center of the domain)
X_common, Y_common = np.meshgrid(np.linspace(0, L, N-1), np.linspace(0, L, N-1))

u_interp = 0.5 * (u[:, :-2] + u[:, 1:-1])  # Average to fit the common grid
v_interp = 0.5 * (v[:-2, :] + v[1:-1, :])  # Average to fit the common grid

# Plot streamlines on the u plot
ax[0].streamplot(X_common, Y_common, u_interp, v_interp, color='k')

plt.show()