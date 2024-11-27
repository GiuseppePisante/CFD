import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
L = 3                    # length of the bar (m)
cp = 897                 # specific heat (J/kg*K)
rho = 2700               # density (kg/m^3)
k = 237                  # thermal conductivity (W/m*K)
alpha = k / (rho * cp)   # thermal diffusivity (m^2/s)

# Apply boundary conditions
T_left = 250     # temperature at the left boundary (K)
T_right = 500    # temperature at the right boundary (K)


# Part (c) - Discretization using a second-order CDS

def CDS_second_derivative(f_values, dx):
    d2f_numerical = np.zeros_like(f_values)

    for i in range(1, len(f_values) - 1):
        d2f_numerical[i] = (f_values[i+1] - 2 * f_values[i] + f_values[i-1]) / dx**2
    return d2f_numerical


# Part (d) - Verification of the correctness of the CDS

# Define the functions and their analytical second derivatives
functions = {
    "f(x) = x": (lambda x: x, lambda x: 0),
    "f(x) = x^2": (lambda x: x**2, lambda x: 2),
    "f(x) = x^3": (lambda x: x**3, lambda x: 6*x),
    "f(x) = sin(x)": (lambda x: np.sin(x), lambda x: -np.sin(x))
}

# Define different grid spacings for convergence study
grid_spacings = [0.1, 0.05, 0.025, 0.0125] 

# Loop over functions to test
for func_name, (f, d2f_analytical) in functions.items():
    errors = []
    
    for dx in grid_spacings:

        # Set up spatial grid
        Nx = int(L / dx) + 1
        x = np.linspace(0, L, Nx)
        
        # Compute function values
        f_values = f(x)
        
        # Analytical second derivative
        d2f_analytical_values = d2f_analytical(x)
        
        # Numerical second derivative 
        d2f_numerical = CDS_second_derivative(f_values, dx)
        
        # Compute error at each point and the L2 norm
        error = d2f_numerical - d2f_analytical_values
        L2_norm_error = np.sqrt(np.sum(error[1:-1]**2) * dx / L)
        errors.append(L2_norm_error)

    # Plot the L2 error norm against grid spacing in a log-log plot
    plt.loglog(grid_spacings, errors, label=func_name, marker='o')

# Plot formatting
plt.xlabel("Grid spacing (Δx)")
plt.ylabel("L2 norm of error")
plt.title("Convergence of Second Derivative Approximation")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()


# Part (e) - Stationary temperature distribution without heat source

# Discretization
h = 0.01                  # grid spacing in meters
N = int(L / h) + 1        # number of grid points
x = np.linspace(0, L, N)  # grid points

# Set up the linear system
A = np.zeros((N, N))
b = np.zeros(N)

# Interior points
for i in range(1, N - 1):
    A[i, i-1] = 1
    A[i, i] = -2
    A[i, i+1] = 1

# Boundary conditions
A[0, 0] = 1
A[-1, -1] = 1
b[0] = T_left
b[-1] = T_right

# Solve the linear system
T_stationary = np.linalg.solve(A, b)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, T_stationary, label="Steady-State Temperature (No Heat Source)")
plt.xlabel("Position along the bar (m)")
plt.ylabel("Temperature (K)")
plt.title("Steady-State Temperature Distribution (No Heat Source)")
plt.grid(True)
plt.legend()
plt.show()


# Part (f) - Stationary temperature distribution with a heat source

# Heat source parameters
q = 200000            # homogeneous heat source (W/m^3)
q_start = int(1 / h)  # start of heat source (m)
q_end = int(2 / h)    # end of heat source (m)

# Interior points
for i in range(1, N - 1):
    A[i, i-1] = 1
    A[i, i] = -2
    A[i, i+1] = 1

    if q_start <= i < q_end:
        b[i] = -q * h**2 / k

# Solve the system
T_stationary_heat_source = np.linalg.solve(A, b)

# Plot results 
plt.figure(figsize=(10, 6))
plt.plot(x, T_stationary_heat_source, label="Steady-State Temperature (With Heat Source)")
plt.xlabel("Position along the bar (m)")
plt.ylabel("Temperature (K)")
plt.title("Steady-State Temperature Distribution (With Heat Source)")
plt.grid(True)
plt.legend()
plt.show()


# Part (g) - Unsteady temperature evolution with heat source

# Time-stepping parameters
dt = 0.01                 # time step (s)
total_time = 1000         # total simulation time (s)
Nt = int(total_time / dt) # number of time steps

# Initialize temperature profile
T = np.linspace(T_left, T_right, N)
T_history = [T.copy()]

T[0] = T_left
T[-1] = T_right

# Define heat source region
source_term = np.zeros(N)
source_term[q_start:q_end] = q / (rho * cp)  # contribution of heat source

# Von Neumann stability criterion
if dt > h**2 / (2 * alpha):
    raise ValueError("Time step is too large")

# Explicit Euler time-stepping with heat source
for n in range(Nt):
    T_new = T.copy()

    for i in range(1, N - 1):
        T_new[i] = T_new[i] + alpha * dt / h**2 * (T_new[i+1] - 2*T_new[i] + T_new[i-1]) + source_term[i] * dt

    T_new[0] = T_left
    T_new[-1] = T_right

    T = T_new.copy()
    T_history.append(T.copy())

# Plot results
plt.figure(figsize=(10, 6))
for i in range(0, Nt, Nt // 10):  
    plt.plot(x, T_history[i], label=f"t={i*dt:.2f} s")
plt.xlabel("Position along the bar (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Evolution Over Time (With Heat Source)")
plt.legend()
plt.grid(True)
plt.show()