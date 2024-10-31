import numpy as np
import matplotlib.pyplot as plt

# Definition of constants
U_inf = 1.0  # (m/s)
V_w = 0.1    # (m/s)
nu = 1.5e-5  # (m^2/s)
rho = 1.225  # (kg/m^3)
p_inf = 101325  # (Pa)

# Calculate boundary layer thickness delta
delta = nu / V_w

# Define the y range
y = np.linspace(0, 0.01, 500)

# Velocity profile
u = U_inf * (1 - np.exp(-y / delta))

# Wall shear stress
tau_wall = rho * nu * (U_inf / delta)

# Friction coefficient
cf = 2 * tau_wall / (rho * U_inf**2)

# Pressure distribution using Bernoulli's equation
p = p_inf + 0.5 * rho * (U_inf**2 - (u**2 + V_w**2))

# Pressure coefficient
cp = 2 * (p - p_inf) / (rho * U_inf**2)

# Print results
print(f"Boundary layer thickness (delta): {delta:.6f} m")
print(f"Wall shear stress (tau_wall): {tau_wall:.6f} N/m^2")
print(f"Friction coefficient (cf): {cf:.6f}")
print(f"Pressure coefficient (cp): {cp[0]:.6f} at the wall")

# Plot the velocity profile
plt.figure(figsize=(8, 6))
plt.plot(u, y, label='Velocity profile')
plt.xlabel('Velocity (u) [m/s]')
plt.ylabel('Distance from the wall (y) [m]')
plt.title('Asymptotic Suction Boundary Layer Velocity Profile')
plt.legend()
plt.grid(True)
plt.savefig('output.png')
plt.close()

# Plot the pressure distribution
plt.figure(figsize=(8, 6))
plt.plot(p, y, label='Pressure distribution')
plt.xlabel('Pressure (p) [Pa]')
plt.ylabel('Distance from the wall (y) [m]')
plt.title('Asymptotic Suction Boundary Layer Pressure Distribution')
plt.legend()
plt.grid(True)
plt.savefig('asymptotic_suction_boundary_layer_pressure.png')
plt.close()