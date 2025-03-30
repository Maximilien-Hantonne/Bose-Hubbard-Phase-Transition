import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import textwrap
import os
import scienceplots 


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


################## PHASE MAP ##################


# Read the data from the file
with open('phase_pres.txt', 'r') as file:
    fixed_param_line = file.readline().strip().split()
    fixed_param = fixed_param_line[0]
    fixed_value = float(fixed_param_line[1])
    data = np.loadtxt(file)

# Extract mu, U, gap_ratio, boson_density, and compressibility
if fixed_param == "J":
    x_label = 'Normalized Interaction Strength ($U/J$)'
    y_label = 'Normalized Chemical Potential ($\mu/J$)'
    x_values = data[:, 0] / fixed_value
    y_values = data[:, 1] / fixed_value
    non_fixed_param1 = 'U'
    non_fixed_param2 = 'mu'
elif fixed_param == "U":
    x_label = 'Normalized Hopping Parameter ($J/U$)'
    y_label = 'Normalized Chemical Potential ($\mu / U$)'
    x_values = data[:, 0] / fixed_value
    y_values = data[:, 1] / fixed_value
    non_fixed_param1 = 'J'
    non_fixed_param2 = 'mu'
elif fixed_param == "u":
    x_label = 'Normalized Hopping Parameter ($J/ \mu$)'
    y_label = 'Normalized Interaction Strength ($U/ \mu$)'
    x_values = data[:, 0] / fixed_value
    y_values = data[:, 1] / fixed_value
    non_fixed_param1 = 'J'
    non_fixed_param2 = 'U'
else:
    raise ValueError("Invalid fixed parameter in phase.txt")

# Calculate min and max values for non-fixed parameters
param1_min = np.min(data[:, 0])
param1_max = np.max(data[:, 0])
param2_min = np.min(data[:, 1])
param2_max = np.max(data[:, 1])

# Extract the gap ratio, boson density, and compressibility
gap_ratio = data[:, 2]
boson_density = data[:, 3]
compressibility = data[:, 4]
# fc = data[:,5]

# Create a grid for x and y
x_unique = np.unique(x_values)
y_unique = np.unique(y_values)
x_grid, y_grid = np.meshgrid(x_unique, y_unique)

# Function to wrap long titles
def wrap_title(title, width=30):
    return "\n".join(textwrap.wrap(title, width))

# Create the directory to save the plots
output_dir = f'../figures/exact/{fixed_param}_{fixed_value}_{non_fixed_param1}_{param1_min}-{param1_max}_{non_fixed_param2}_{param2_min}-{param2_max}'
os.makedirs(output_dir, exist_ok=True)

# Plot the heatmap for gap ratio
fig1 = plt.figure(figsize=(11, 11))

contour1 = plt.contourf(x_grid, y_grid, gap_ratio.reshape(len(y_unique), len(x_unique)), levels=50, cmap='viridis')
cbar1 = plt.colorbar(contour1, label='Gap Ratio')
cbar1.ax.tick_params(labelsize=20)
cbar1.ax.axhline(y=0.39, color='red', linestyle='solid', linewidth=3)
cbar1.ax.axhline(y=0.53, color='red', linestyle='solid', linewidth=3)
plt.tick_params(axis='both', which='major', labelsize=25)  # Increase tick label size
plt.xlabel(x_label, fontsize = 35)
plt.ylabel(y_label, fontsize = 35)
cbar1.set_label(r'Gap ratio', fontsize=35)

# Save the plot
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, "..", "figures")
plt.savefig(os.path.join(save_path, 'gap_ratio_pres.svg'))

plt.tight_layout()
plt.show()
plt.close(fig1)