import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import scienceplots 
import os

# Enable LaTex rendering in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, "../figures/mean_field")

# Science plots style
# plt.style.use(['science', 'grid'])

# Read the data from the file
with open(os.path.join(current_dir, "build","mean_field.txt"), "r") as file:
    data = np.loadtxt(file)


#Extract mu, J, and Psi the superfluid density 
mus = data[:,0]
Js = data[:,1]
Psis = data[:,2]
Psis = np.where(Psis < 0, 0, Psis)

mus_unique = np.unique(mus)
Js_unique = np.unique(Js)
mus_grid, Js_grid = np.meshgrid(mus_unique, Js_unique)

fig = plt.figure(figsize=(10, 10))
contour = plt.contourf(Js_grid, mus_grid, Psis.reshape(len(mus_unique), len(Js_unique)), cmap='viridis')
cbar = plt.colorbar(contour, label=r'$\Psi$')
cbar.ax.tick_params(labelsize=14)  # Increase colorbar tick label size
cbar.set_label(r'$\Psi$', fontsize=16)  # Increase colorbar label size

plt.xlabel(r'$\frac{qJ}{U}$', fontsize=16)  # Increase x-axis label size
plt.ylabel(r'$\frac{\mu}{U}$', fontsize=16)  # Increase y-axis label size
plt.ylim(0, 2.5)
plt.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
plt.ylim(0, 2.5)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "mean_field_test2.png"))
plt.close(fig)