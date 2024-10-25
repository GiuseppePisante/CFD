import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt

# Define the Blasius equation as a system of first-order ODEs
def blasius(eta, y):
    f, f_prime, f_double_prime = y
    return [f_prime, f_double_prime, -0.5 * f * f_double_prime]

# Define initial conditions
initial_conditions = [0, 0, 1.5399]  # f(0) = 0, f'(0) = 0, f''(0) = 1.5399

# Define the range for the independent variable (eta)
eta_range = (0, 10)

# Solve the ODE system
solution = solve_ivp(blasius, eta_range, initial_conditions, t_eval=np.linspace(0, 10, 500))

# Extract the results
eta = solution.t
f = solution.y[0]
f_prime = solution.y[1]
f_double_prime = solution.y[2]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(eta, f, label="f(η)")
plt.plot(eta, f_prime, label="f'(η)")
plt.plot(eta, f_double_prime, label="f''(η)")
plt.xlabel("η")
plt.ylabel("Function values")
plt.title("Solution to the Blasius Equation")
plt.legend()
plt.grid(True)
plt.savefig('output_plot.png')
plt.close()

# Calculate the wall shear stress (τ_wall)
mu = 1.0  # Dynamic viscosity (example value)
U_inf = 1.0  # Free-stream velocity (example value)
rho = 1.0  # Fluid density (example value)
tau_wall = mu * f_double_prime[0]

# Calculate the friction coefficient (c_f)
c_f = 2 * tau_wall / (rho * U_inf**2)

# Calculate the pressure coefficient (c_p)
p_wall = 1.0  # Pressure at the wall (example value)
p_inf = 1.0  # Free-stream pressure (example value)
c_p = 2 * (p_wall - p_inf) / (rho * U_inf**2)

# Calculate the boundary layer displacement thickness (δ*)
x = 1.0  # Distance from the leading edge (example value)
nu = 1.0  # Kinematic viscosity (example value)
Re_x = U_inf * x / nu
delta_star = (1.7208 * x) / np.sqrt(Re_x)

# Print the results
print(f"Wall shear stress (τ_wall): {tau_wall}")
print(f"Friction coefficient (c_f): {c_f}")
print(f"Pressure coefficient (c_p): {c_p}")
print(f"Boundary layer displacement thickness (δ*): {delta_star}")

# Explanation of the significance of δ*
print("\nSignificance of the displacement thickness (δ*):")
print("The displacement thickness represents the distance by which the outer potential flow is displaced due to the boundary layer.")
print("It accounts for the reduction in flow rate caused by the presence of the boundary layer.")
print("In this setup, δ* depends on the free-stream velocity (U_inf), the kinematic viscosity (ν), and the distance from the leading edge (x).")
print("As U_inf increases, δ* decreases, indicating a thinner boundary layer. Conversely, as ν or x increases, δ* increases, indicating a thicker boundary layer.")