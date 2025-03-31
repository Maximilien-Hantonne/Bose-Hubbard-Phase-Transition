import matplotlib
matplotlib.use('TkAgg')

import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scienceplots 


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


################## PHASE MAP ##################

# Read the data from the file
with open('phase.txt', 'r') as file:
    fixed_param_line = file.readline().strip().split()
    fixed_param = fixed_param_line[0]
    fixed_value = float(fixed_param_line[1])
    data = np.loadtxt(file)

# Extract J, mu, U, gap_ratio, condensate fraction, and coherence
if fixed_param == "J":
    x_label = 'Interaction Strength (U)'
    y_label = 'Chemical Potential (mu)'
    x_values = data[:, 0] 
    y_values = data[:, 1] 
    non_fixed_param1 = 'U'
    non_fixed_param2 = 'mu'
elif fixed_param == "U":
    x_label = 'Hopping Parameter (J)'
    y_label = 'Chemical Potential (mu)'
    x_values = data[:, 0] 
    y_values = data[:, 1] 
    non_fixed_param1 = 'J'
    non_fixed_param2 = 'mu'
elif fixed_param == "u":
    x_label = 'Hopping Parameter (J)'
    y_label = 'Interaction Strength (U)'
    x_values = data[:, 0] 
    y_values = data[:, 1] 
    non_fixed_param1 = 'J'
    non_fixed_param2 = 'U'
else:
    raise ValueError("Invalid fixed parameter in phase.txt")

# Calculate min and max values for non-fixed parameters
param1_min = np.min(data[:, 0])
param1_max = np.max(data[:, 0])
param2_min = np.min(data[:, 1])
param2_max = np.max(data[:, 1])

# Extract the gap ratio, condensate fraction, and coherence
gap_ratio = data[:,2]
condensate_fraction = data[:, 3]
coherence = data[:, 4]

# Create a grid for x and y
x_unique = np.unique(x_values)
y_unique = np.unique(y_values)
x_grid, y_grid = np.meshgrid(x_unique, y_unique)

# Reshape the data arrays to match the grid shape (flatten them for later reshaping)
gap_ratio_grid = gap_ratio.reshape(len(y_unique), len(x_unique))
condensate_fraction_grid = condensate_fraction.reshape(len(y_unique), len(x_unique))
coherence_grid = coherence.reshape(len(y_unique), len(x_unique))

# Apply Gaussian blur for better smoothing (lissage)
sigma = 2  # Adjust the value of sigma for more/less blur
gap_ratio_blurred = gaussian_filter(gap_ratio_grid, sigma=sigma)
condensate_fraction_blurred = gaussian_filter(condensate_fraction_grid, sigma=sigma)
coherence_blurred = gaussian_filter(coherence_grid, sigma=sigma)

# Function to wrap long titles
def wrap_title(title, width=30):
    return "\n".join(textwrap.wrap(title, width))

# Create the directory to save the plots
# output_dir = f'../figures/exact/{fixed_param}_{fixed_value}_{non_fixed_param1}_{param1_min}-{param1_max}_{non_fixed_param2}_{param2_min}-{param2_max}'
# os.makedirs(output_dir, exist_ok=True)

# Plot the heatmap for gap ratio
fig1 = plt.figure(figsize=(10, 10))

contour1 = plt.contourf(x_grid, y_grid, gap_ratio_blurred, levels=50, cmap='viridis')
cbar1 = plt.colorbar(contour1, label='Gap Ratio')
cbar1.ax.tick_params(labelsize=20)
cbar1.ax.axhline(y=0.39, color='red', linestyle='solid', linewidth=3)
cbar1.ax.axhline(y=0.53, color='red', linestyle='solid', linewidth=3)
plt.tick_params(axis='both', which='major', labelsize=25)  # Increase tick label size
plt.xlabel(x_label, fontsize = 35)
plt.ylabel(y_label, fontsize = 35)
cbar1.set_label(r'Gap ratio', fontsize=35)
plt.tight_layout()
# plt.title(wrap_title('Gap Ratio with respect to {} and {}'.format(x_label, y_label)), fontsize=18)
# plt.figtext(0.5, 0.01, 'Note: 0.39 is for a Poissonnian distribution and 0.53 is for a Gaussian orthogonal ensemble (GOE)', ha='center', fontsize=9, color='red')

# Save the plot
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, "..", "figures")
plt.savefig(os.path.join(save_path, 'gap_ratio.svg'))
plt.close(fig1)


# Plot and save the condensate fraction plot
fig2 = plt.figure(figsize=(10, 10))
contour2 = plt.contourf(x_grid, y_grid, condensate_fraction_blurred, levels=50, cmap='viridis')
cbar2 = plt.colorbar()
plt.tick_params(axis='both', which='major', labelsize=25)  # Increase tick label size
plt.xlabel(x_label, fontsize = 35)
plt.ylabel(y_label, fontsize = 35)
cbar2.set_label(r'Condensate fraction', fontsize=35)
cbar2.ax.tick_params(labelsize=20)
plt.title(wrap_title('Condensate fraction with respect to {} and {}'.format(x_label, y_label)), fontsize=18)
plt.savefig(os.path.join(save_path, 'condensate_fraction_plot.svg'))  # Save as SVG
plt.close(fig2)

# Plot and save the coherence plot
fig3 = plt.figure(figsize=(10, 10))
plt.contourf(x_grid, y_grid, coherence_blurred, levels=50, cmap='viridis')
cbar3 = plt.colorbar()
plt.tick_params(axis='both', which='major', labelsize=25)  # Increase tick label size
plt.xlabel(x_label, fontsize = 35)
plt.ylabel(y_label, fontsize = 35)
cbar3.set_label(r'Cohrence in boson density', fontsize=35)
cbar3.ax.tick_params(labelsize=20)
plt.title(wrap_title('Coherence with respect to {} and {}'.format(x_label, y_label)), fontsize=18)

# Save the plot
plt.savefig(os.path.join(save_path, 'coherence_plot.svg'))

plt.tight_layout()
plt.close(fig3)