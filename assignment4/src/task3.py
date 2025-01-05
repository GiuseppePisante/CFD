import numpy as np
import matplotlib.pyplot as plt

# Define the range of Courant numbers (c)
c = np.linspace(0, 1, 500)  # c ranges from 0 to 1

# Stability curve: d = c^2 / 2
d_curve = c**2 / 2

# Upper limit for d: d <= 1/2
d_max = 0.5

# Plot the curve
plt.figure(figsize=(8, 6))
plt.plot(c, d_curve, label=r'$d = \frac{c^2}{2}$', color='blue', linewidth=2)
plt.axhline(d_max, color='red', linestyle='--', label=r'$d = \frac{1}{2}$')

# Shade the stability region
plt.fill_between(c, 0, np.minimum(d_curve, d_max), color='lightblue', alpha=0.5)

# Labels and title
plt.xlabel(r'Courant Number $c$', fontsize=14)
plt.ylabel(r'Diffusion Number $d$', fontsize=14)
plt.title('Stability Region for the Unsteady Advection-Diffusion Equation', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Save the plot as a figure
plt.savefig('/home/giuseppepisante/FAU/CFD/assignment4/src/stability_region_plot.png')
# Show the plot
plt.show()
