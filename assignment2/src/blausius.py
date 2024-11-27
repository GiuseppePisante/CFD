import numpy as np
import matplotlib.pyplot as plt

# Definition of the Blausius system
def blasius_system(eta, f):
    f1, f2, f3 = f  
    df1_deta = f2
    df2_deta = f3
    df3_deta = -0.5 * f1 * f3
    return np.array([df1_deta, df2_deta, df3_deta])

# Explicit Euler method 
def euler_method(f, eta_max, delta_eta):
    eta_values = np.arange(0, eta_max + delta_eta, delta_eta)
    f_values = np.zeros((len(eta_values), 3))
    f_values[0] = f  # Initial conditions
    
    for i in range(1, len(eta_values)):
        f_values[i] = f_values[i - 1] + delta_eta * blasius_system(eta_values[i - 1], f_values[i - 1])
    
    return eta_values, f_values

# Runge-Kutta 4 (RK4) method 
def rk4_method(f, eta_max, delta_eta):
    eta_values = np.arange(0, eta_max + delta_eta, delta_eta)
    f_values = np.zeros((len(eta_values), 3))
    f_values[0] = f  # Initial conditions

    for i in range(1, len(eta_values)):
        eta = eta_values[i - 1]
        f_current = f_values[i - 1]
        
        k1 = delta_eta * blasius_system(eta, f_current)
        k2 = delta_eta * blasius_system(eta + delta_eta / 2, f_current + k1 / 2)
        k3 = delta_eta * blasius_system(eta + delta_eta / 2, f_current + k2 / 2)
        k4 = delta_eta * blasius_system(eta + delta_eta, f_current + k3)
        
        f_values[i] = f_current + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return eta_values, f_values

# Initial conditions
initial_conditions = np.array([0, 0, 0.33])  # value for f''(0) from website quoted in the report

# Define parameters
eta_max = 8.0
step_sizes = [0.2, 0.05]

# Solve using the Explicit Euler method
euler_solutions = {}
for delta_eta in step_sizes:
    eta_values, f_values = euler_method(initial_conditions, eta_max, delta_eta)
    euler_solutions[delta_eta] = (eta_values, f_values)

# Solve using the RK4 method
rk4_solutions = {}
for delta_eta in step_sizes:
    eta_values, f_values = rk4_method(initial_conditions, eta_max, delta_eta)
    rk4_solutions[delta_eta] = (eta_values, f_values)


# (b) Show the numerical solutions for f, f', and f''
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
titles = ["$f(\\eta)$", "$f'(\\eta)$", "$f''(\\eta)$"]

# Plot for Euler method results
for i in range(3):
    for delta_eta, (eta_values, f_values) in euler_solutions.items():
        axs[i].plot(eta_values, f_values[:, i], label=f"Euler, $\\Delta \\eta = {delta_eta}$")
        
# Plot for RK4 method results
for i in range(3):
    for delta_eta, (eta_values, f_values) in rk4_solutions.items():
        axs[i].plot(eta_values, f_values[:, i], label=f"RK4, $\\Delta \\eta = {delta_eta}$")
    
    axs[i].set_title(titles[i])
    axs[i].set_xlabel("$\\eta$")
    axs[i].set_ylabel(titles[i])
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()


# (c) Velocity components u* and v* using RK4 method with delta_eta = 0.05
eta_values_rk4, f_values_rk4 = rk4_solutions[0.05]
u_star = f_values_rk4[:, 1]  # u* = f'(\eta)
v_star = 0.5 * eta_values_rk4 * f_values_rk4[:, 1] - 0.5 * f_values_rk4[:, 0]  # v*

# Plot u* and v*
plt.figure(figsize=(10, 6))
plt.plot(eta_values_rk4, u_star, label="$u^* = f'(\\eta)$")
plt.plot(eta_values_rk4, v_star, label="$v^* = \\frac{1}{2}(\\eta f'(\\eta) - f(\\eta))$")
plt.xlabel("$\\eta$")
plt.ylabel("Velocity Components")
plt.title("Velocity Profiles $u^*$ and $v^*$")
plt.legend()
plt.grid()
plt.show()