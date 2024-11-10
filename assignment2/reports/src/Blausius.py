import numpy as np
import matplotlib.pyplot as plt

# Blasius equation as a system of first-order ODEs
def blasius_system(eta, f):
    f1, f2, f3 = f  
    df1_deta = f2
    df2_deta = f3
    df3_deta = -0.5 * f1 * f3
    return np.array([df1_deta, df2_deta, df3_deta])

# Euler method for solving the system
def euler_method(f, eta_max, delta_eta):
    eta_values = np.arange(0, eta_max + delta_eta, delta_eta)
    f_values = np.zeros((len(eta_values), 3))
    f_values[0] = f  # Initial conditions
    
    for i in range(1, len(eta_values)):
        f_values[i] = f_values[i - 1] + delta_eta * blasius_system(eta_values[i - 1], f_values[i - 1])
    
    return eta_values, f_values

# Runge-Kutta 4 (RK4) method for solving the system
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
initial_conditions = np.array([0, 0, 0.33206])  # Guess value for f''(0) to start iteration

# Define parameters
eta_max = 8.0
step_sizes = [0.2, 0.05]

# Solve using both methods and both step sizes
solutions = {}
for method, method_name in zip([euler_method, rk4_method], ["Euler", "RK4"]):
    for delta_eta in step_sizes:
        eta_values, f_values = method(initial_conditions, eta_max, delta_eta)
        solutions[(method_name, delta_eta)] = (eta_values, f_values)

# (b) Show the numerical solutions for f, f', and f''
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
titles = ["$f(\\eta)$", "$f'(\\eta)$", "$f''(\\eta)$"]

for i in range(3):
    for (method_name, delta_eta), (eta_values, f_values) in solutions.items():
        axs[i].plot(eta_values, f_values[:, i], label=f"{method_name}, $\\Delta \\eta = {delta_eta}$")
    axs[i].set_title(titles[i])
    axs[i].set_xlabel("$\\eta$")
    axs[i].set_ylabel(titles[i])
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()

# (c) Calculate and Plot the Velocity Profiles u* and v*
# Extract the RK4 solution with delta_eta = 0.05
eta_values_rk4, f_values_rk4 = solutions[("RK4", 0.05)]

# Calculate the velocity profiles
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

# (d) Extract and Report the Value of f''(0)
# Use the first value of f'' in the RK4 solution with delta_eta = 0.05
fpp_0 = f_values_rk4[0, 2]
print("The value of f''(0) from the RK4 solution is:", fpp_0)