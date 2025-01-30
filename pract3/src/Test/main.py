import matplotlib.pyplot as plt
import numpy as np
import solver
from matplotlib import cm


L=1
N=[50,100,200]
RE=[1,10,1000]
u_list=[]
v_list=[]
p_list=[]

dict_Re={}

for Re in RE:
    for i in N:
      key = Re+i
      u,v,p=solver.solver(i,L,Re)
      u_list.append(u)
      v_list.append(v)
      p_list.append(p)
      dict_Re[key]=[u,v,p]

# h = L/(N-1)
# x = np.linspace(0,L,N)
# y = np.linspace(0,L,N)
# y_dom = 1 - ((np.arange(1, N + 1) - 1) * h)

# X, Y = np.meshgrid(x, y_dom)
# X1,Y1=np.meshgrid(x,y)
# u_reversed_rows = u[::-1]
# v_reversed_rows = v[::-1]
# p_reversed_rows = p[::-1]
# vorticity_list=[]
# for i in range(len(u_list)):
#   Uy, Ux = np.gradient(u_list[i],h,h)
#   Vy, Vx = np.gradient(v_list[i],h,h)
#   vorticity = Vx - Uy
#   vorticity_list.append(vorticity)
# fig, axes = plt.subplots(5, 3, figsize=(12, 15))
# fig.suptitle("Flow Visualization with U and V velocity, Pressure, Streamlines and Vorticity")
# for j in range(3):
#         ax = axes[0,j]
#         contour = ax.contourf(X, Y, u_list[j], levels=20, alpha=0.5, cmap=cm.plasma)
#         ax.set_title(f"U velocity, Re = {RE[j]}")
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         cbar = plt.colorbar(contour, ax=ax)
#         cbar.set_label('U velocity')
#         ax = axes[1,j]
#         contour = ax.contourf(X, Y, v_list[j],levels=20,alpha=0.5, cmap=cm.plasma)
#         ax.set_title(f"V velocity, Re = {RE[j]}")
#         cbar = plt.colorbar(contour, ax=ax)
#         cbar.set_label('V velocity')  
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax = axes[2,j]
#         contour= ax.contourf(X, Y, p_list[j],levels=20,alpha=0.5, cmap=cm.plasma)
#         ax.set_title(f"Pressure, Re = {RE[j]}")
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         cbar = plt.colorbar(contour, ax=ax)
#         cbar.set_label('Pressure')
#         ax = axes[3,j]
#         ax.streamplot(X1,Y1, u_list[j][::-1], v_list[j][::-1] ,cmap=cm.plasma, linewidth=2)
#         ax.set_title(f"Streamlines, Re = {RE[j]}")
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax = axes[4, j]
#         ax.contour(X, Y, vorticity_list[j],levels=500)
#         ax.set_title(f"Vorticity field, Re = {RE[j]}")
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')

# # # Adjust layout
# plt.tight_layout()
# fig.savefig("Task4.png")



fig, axes = plt.subplots(3, 3, figsize=(12, 12))
RE=[1,10,1000]

fig.suptitle("Grid study with 50,100,200 at horizontal plane at center of vortex")
x1 = np.linspace(0,L,50)
x2 = np.linspace(0,L,100)
x3 = np.linspace(0,L,200)

for i in range(len(RE)):
    for j in range(len(N)):
        ax= axes[i,j]
        if j==0:
            labelstr="U"
            ax.set_ylabel('U')
        elif j==1:
            labelstr="V"
            ax.set_ylabel('V')
        if j==2:
            labelstr="P"
            ax.set_ylabel('P')
        if RE[i]==1 or RE[i]==10:
            ax.plot(x1,dict_Re[RE[i]+N[0]][j][:,40],label=f"{labelstr}, N={N[0]}")
            ax.plot(x2,dict_Re[RE[i]+N[1]][j][:,80],label=f"{labelstr}, N={N[1]}")
            ax.plot(x3,dict_Re[RE[i]+N[2]][j][:,160],label=f"{labelstr}, N={N[2]}")
        else:    
            ax.plot(x1,dict_Re[RE[i]+N[0]][j][:,30],label=f"{labelstr}, N={N[0]}")
            ax.plot(x2,dict_Re[RE[i]+N[1]][j][:,60],label=f"{labelstr}, N={N[1]}")
            ax.plot(x3,dict_Re[RE[i]+N[2]][j][:,120],label=f"{labelstr}, N={N[2]}")
        ax.set_title(f"Reynolds Number Re = {RE[i]}")
        ax.set_xlabel('x')
        ax.legend()
plt.tight_layout()
fig.savefig("horizontal_plane.png")


fig.suptitle("Grid study with 50,100,200 at vertical plane x=0.5")
x1 = np.linspace(0,L,50)
x2 = np.linspace(0,L,100)
x3 = np.linspace(0,L,200)
for i in range(len(RE)):
    for j in range(len(N)):
        ax= axes[i,j]
        if j==0:
            labelstr="U"
            ax.set_ylabel('U')
        elif j==1:
            labelstr="V"
            ax.set_ylabel('V')
        if j==2:
            labelstr="P"
            ax.set_ylabel('P')
        ax.plot(x1,dict_Re[RE[i]+N[0]][j][int(N[0]/2),:],label=f"{labelstr}, N={N[0]}")
        ax.plot(x2,dict_Re[RE[i]+N[1]][j][int(N[1]/2),:],label=f"{labelstr}, N={N[1]}")
        ax.plot(x3,dict_Re[RE[i]+N[2]][j][int(N[2]/2),:],label=f"{labelstr}, N={N[2]}")

        ax.set_title(f"Reynolds Number Re = {RE[i]}")
        ax.set_xlabel('y')
        ax.legend()
plt.tight_layout()
fig.savefig("Vertical_plane.png")
