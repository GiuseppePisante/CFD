import matplotlib.pyplot as plt
import numpy as np
from solver import * 
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator

L = 1
N = [20, 30, 60, 90]
RE = [1000]

u_list = []
v_list = []
p_list = []

dict_Re = {}

def interpolate_to_smaller(U_large, N_large, N_small):
    """Interpolate a larger grid solution to match a smaller grid using RegularGridInterpolator."""
    x_large = np.linspace(0, L, N_large)
    y_large = np.linspace(0, L, N_large)
    x_small = np.linspace(0, L, N_small)
    y_small = np.linspace(0, L, N_small)

    interpolator = RegularGridInterpolator((x_large, y_large), U_large, method='linear')
    
    # Creiamo le coordinate della griglia più piccola per l'interpolazione
    X_small, Y_small = np.meshgrid(x_small, y_small, indexing='ij')
    points = np.array([X_small.ravel(), Y_small.ravel()]).T
    
    # Interpoliamo e rimappiamo alla forma originale
    U_small = interpolator(points).reshape(N_small, N_small)
    
    return U_small

for Re in RE:
    for i in N:
        key = Re + i
        u, v, p = solver(i, L, Re)
        u_list.append(u)
        v_list.append(v)
        p_list.append(p)
        dict_Re[key] = [u, v, p]

    # Interpolazione per il confronto
    u_interp_1 = interpolate_to_smaller(u_list[2], N[2], N[0]) 
    u_interp_2 = interpolate_to_smaller(u_list[3], N[3], N[1])

    u_diff_1 = u_list[0] - u_interp_1
    u_diff_2 = u_list[1] - u_interp_2

    # Plotting the differences
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the difference between u of N[0] and u of N[2] (Interpolated)
    ax = axes[0]
    contour = ax.contourf(np.linspace(0, L, N[0]), np.linspace(0, L, N[0]), u_diff_1, levels=200, cmap=cm.viridis)
    ax.set_title("Difference between u(N=50) and u(N=150) (Interpolated)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Difference in u')

    # Plot the difference between u of N[1] and u of N[3] (Interpolated)
    ax = axes[1]
    contour = ax.contourf(np.linspace(0, L, N[1]), np.linspace(0, L, N[1]), u_diff_2, levels=200, cmap=cm.viridis)
    ax.set_title("Difference between u(N=100) and u(N=300) (Interpolated)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Difference in u')

    plt.tight_layout()
    plt.show()

    # Interpolazione per il confronto
    p_interp_1 = interpolate_to_smaller(p_list[2], N[2], N[0])  # p(N=30) → N=10
    p_interp_2 = interpolate_to_smaller(p_list[3], N[3], N[1])  # p(N=45) → N=15

    p_diff_1 = p_list[0] - p_interp_1
    p_diff_2 = p_list[1] - p_interp_2

    # Plotting the differences
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the difference between p of N[0] and p of N[2] (Interpolated)
    ax = axes[0]
    contour = ax.contourf(np.linspace(0, L, N[0]), np.linspace(0, L, N[0]), p_diff_1, levels=200, cmap=cm.viridis)
    ax.set_title("Difference between p(N=50) and p(N=150) (Interpolated)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Difference in p')

    # Plot the difference between p of N[1] and p of N[3] (Interpolated)
    ax = axes[1]
    contour = ax.contourf(np.linspace(0, L, N[1]), np.linspace(0, L, N[1]), p_diff_2, levels=200, cmap=cm.viridis)
    ax.set_title("Difference between p(N=100) and p(N=300) (Interpolated)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Difference in p')

    plt.tight_layout()
    plt.show()

# fig, axes = plt.subplots(3, 3, figsize=(12, 12))
# RE=[1,10,1000]

# fig.suptitle("Grid study with 50,100,200 at horizontal plane at center of vortex")
# x1 = np.linspace(0,L,N[0])
# x2 = np.linspace(0,L,N[1])
# x3 = np.linspace(0,L,N[2])

# for i in range(len(RE)):
#     for j in range(len(N)):
#         ax= axes[i,j]
#         if j==0:
#             labelstr="U"
#             ax.set_ylabel('U')
#         elif j==1:
#             labelstr="V"
#             ax.set_ylabel('V')
#         if j==2:
#             labelstr="P"
#             ax.set_ylabel('P')
#         if RE[i]==1 or RE[i]==10:
#             x1_smooth, y1_smooth = smooth_curve(x1, dict_Re[RE[i]+N[0]][j][:, int(3/5*N[0])], 0.5, 50)
#             x2_smooth, y2_smooth = smooth_curve(x2, dict_Re[RE[i]+N[1]][j][:, int(3/5*N[1])], 0.5, 100)
#             x3_smooth, y3_smooth = smooth_curve(x3, dict_Re[RE[i]+N[2]][j][:, int(3/5*N[2])], 0.5, 200)

#             # Plot smoothed curves
#             ax.plot(x1_smooth, y1_smooth, label=f"{labelstr}, N=50")
#             ax.plot(x2_smooth, y2_smooth, label=f"{labelstr}, N=100")
#             ax.plot(x3_smooth, y3_smooth, label=f"{labelstr}, N=200")
#         else:
#             x1_smooth, y1_smooth = smooth_curve(x1, dict_Re[RE[i]+N[0]][j][:, int(3/5*N[0])], 0.5, 50)
#             x2_smooth, y2_smooth = smooth_curve(x2, dict_Re[RE[i]+N[1]][j][:, int(3/5*N[1])], 0.5, 100)
#             x3_smooth, y3_smooth = smooth_curve(x3, dict_Re[RE[i]+N[2]][j][:, int(3/5*N[2])], 0.5, 200)

#             # Plot smoothed curves
#             ax.plot(x1_smooth, y1_smooth, label=f"{labelstr}, N=50")
#             ax.plot(x2_smooth, y2_smooth, label=f"{labelstr}, N=100")
#             ax.plot(x3_smooth, y3_smooth, label=f"{labelstr}, N=200")
#         ax.set_title(f"Reynolds Number Re = {RE[i]}")
#         ax.set_xlabel('x')
#         ax.legend()
# plt.tight_layout()
# fig.savefig("horizontal_plane.png")

# fig, axes = plt.subplots(3, 3, figsize=(12, 12))
# fig.suptitle("Grid study with 50,100,200 at vertical plane x=0.5")
# x1 = np.linspace(0,L,N[0])
# x2 = np.linspace(0,L,N[1])
# x3 = np.linspace(0,L,N[2])
# for i in range(len(RE)):
#     for j in range(len(N)):
#         ax= axes[i,j]
#         if j==0:
#             labelstr="U"
#             ax.set_ylabel('U')
#         elif j==1:
#             labelstr="V"
#             ax.set_ylabel('V')
#         if j==2:
#             labelstr="P"
#             ax.set_ylabel('P')
#         x1_smooth, y1_smooth = smooth_curve(x1,dict_Re[RE[i]+N[0]][j][int(N[0]/2),:], 0.5, 50)
#         x2_smooth, y2_smooth = smooth_curve(x2,dict_Re[RE[i]+N[1]][j][int(N[1]/2),:], 0.5, 100)
#         x3_smooth, y3_smooth = smooth_curve(x3,dict_Re[RE[i]+N[2]][j][int(N[2]/2),:], 0.5, 200)

#         # Plot smoothed curves
#         ax.plot(x1_smooth, y1_smooth, label=f"{labelstr}, N=50")
#         ax.plot(x2_smooth, y2_smooth, label=f"{labelstr}, N=100")
#         ax.plot(x3_smooth, y3_smooth, label=f"{labelstr}, N=200")

#         ax.set_title(f"Reynolds Number Re = {RE[i]}")
#         ax.set_xlabel('y')
#         ax.legend()
# plt.tight_layout()
# fig.savefig("Vertical_plane.png")


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

