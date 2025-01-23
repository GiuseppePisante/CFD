from solver import Solver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

L=1
N=100
Re=1000

L=1
N=100
RE=[1,10,1000]
u_list=[]
v_list=[]
p_list=[]

for Re in RE:
    u,v,p=Solver(N,L,Re)
    u_list.append(u)
    v_list.append(v)
    p_list.append(p)

h = L/(N-1)
x = np.linspace(0,L,N)
y = np.linspace(0,L,N)
y_dom = 1 - ((np.arange(1, N + 1) - 1) * h)

X, Y = np.meshgrid(x, y_dom)
X1,Y1=np.meshgrid(x,y)
u_reversed_rows = u[::-1]
v_reversed_rows = v[::-1]
p_reversed_rows = p[::-1]
vorticity_list=[]
for i in range(len(u_list)):
  Uy, Ux = np.gradient(u_list[i],h,h)
  Vy, Vx = np.gradient(v_list[i],h,h)
  vorticity = Vx - Uy
  vorticity_list.append(vorticity)

fig, axes = plt.subplots(5, 3, figsize=(12, 15))
fig.suptitle("Velocity and Pressure fields for different Reynolds numbers")
for j in range(3):
        ax = axes[0, j]
        contour = ax.contourf(X, Y, u_list[j], levels=20, alpha=0.75, cmap='coolwarm')
        ax.set_title(f"u, Re = {RE[j]}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('u')
        
        ax = axes[1, j]
        contour = ax.contourf(X, Y, v_list[j], levels=20, alpha=0.75, cmap='coolwarm')
        ax.set_title(f"v, Re = {RE[j]}")
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('v')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        ax = axes[2, j]
        contour = ax.contourf(X, Y, p_list[j], levels=20, alpha=0.75, cmap='coolwarm')
        ax.set_title(f"Pressure, Re = {RE[j]}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Pressure')
        
        ax = axes[3, j]
        strm = ax.streamplot(X1, Y1, u_list[j][::-1], v_list[j][::-1])
        ax.set_title(f"Streamlines, Re = {RE[j]}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.set_aspect('equal')
        
        ax = axes[4, j]
        ax.contour(X, Y, vorticity_list[j], levels=500, cmap=cm.plasma)
        ax.set_title(f"Vorticity field, Re = {RE[j]}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for j in range(3):
  ax = axes[j]
  ax.quiver(X[::2, ::2], Y[::2, ::2], u_list[j][::2, ::2], v_list[j][::2, ::2], scale=10,color='blue')
  ax.set_title(f"Velocity field, Re = {RE[j]}")
  ax.set_xlabel('x')
  ax.set_ylabel('y')
plt.tight_layout()
