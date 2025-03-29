import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import textwrap
import os


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


################## PHASE MAP ##################


# Read the data from the file
with open('phase.txt', 'r') as file:
    fixed_param_line = file.readline().strip().split()
    fixed_param = fixed_param_line[0]
    fixed_value = float(fixed_param_line[1])
    data = np.loadtxt(file)

# Extract mu, U, gap_ratio, boson_density, and compressibility
if fixed_param == "J":
    x_label = 'Normalized Interaction Strength (U/J)'
    y_label = 'Normalized Chemical Potential (mu/J)'
    x_values = data[:, 0] / fixed_value
    y_values = data[:, 1] / fixed_value
    non_fixed_param1 = 'U'
    non_fixed_param2 = 'mu'
elif fixed_param == "U":
    x_label = 'Normalized Hopping Parameter (J/U)'
    y_label = 'Normalized Chemical Potential (mu/U)'
    x_values = data[:, 0] / fixed_value
    y_values = data[:, 1] / fixed_value
    non_fixed_param1 = 'J'
    non_fixed_param2 = 'mu'
elif fixed_param == "u":
    x_label = 'Normalized Hopping Parameter (J/mu)'
    y_label = 'Normalized Interaction Strength (U/mu)'
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
plt.xlabel(x_label, fontsize = 35)
plt.ylabel(y_label, fontsize = 35)
cbar1.set_label(r'Gap ratio', fontsize=35)

# Save the plot
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, "..", "figures")
plt.savefig(save_path, 'gap_ratio.svg')

plt.tight_layout()
plt.show()
plt.close()



################## PCA ##################


# # Load the PCA results
# pca_matrices = []
# with open('pca_results.csv', 'r') as file:
#     lines = file.readlines()
#     current_matrix = []
#     for line in lines:
#         if line.startswith("PCA"):
#             if current_matrix:
#                 pca_matrices.append(np.array(current_matrix))
#                 current_matrix = []
#         elif line.strip():  # Check if the line is not empty
#             current_matrix.append([float(x) for x in line.strip().split(',')])
#     if current_matrix:
#         pca_matrices.append(np.array(current_matrix))

# # Load the dispersion values
# dispersions = []
# with open('dispersions.csv', 'r') as file:
#     lines = iter(file.readlines())
#     for line in lines:
#         if line.startswith("Dispersion"):
#             try:
#                 dispersion_line = next(lines).strip()
#                 dispersion = float(dispersion_line)
#                 dispersions.append(dispersion)
#             except StopIteration:
#                 break

# # Load the cluster labels
# cluster_labels = []
# with open('cluster_labels.csv', 'r') as file:
#     lines = file.readlines()
#     current_labels = []
#     for line in lines:
#         if line.startswith("Clusters"):
#             if current_labels:
#                 cluster_labels.append(np.array(current_labels))
#                 current_labels = []
#         elif line.strip():  # Check if the line is not empty
#             current_labels.append(int(line.strip()))
#     if current_labels:
#         cluster_labels.append(np.array(current_labels))

# # Create the animation for PCA results
# fig, ax = plt.subplots(figsize=(10, 8))

# def update_pca(frame):
#     ax.clear()
#     projected_data = pca_matrices[frame]
#     clusters = cluster_labels[frame]
#     dispersion = dispersions[frame]
#     scatter = ax.scatter(projected_data[:, 0], projected_data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
#     ax.set_title(f'Projection onto the First Two Principal Components \n (Dispersion: {dispersion:.4f}) \n \n \n', fontsize=12)
#     ax.set_xlabel('Principal Component 1')
#     ax.set_ylabel('Principal Component 2')
#     ax.grid(True)
#     ax.legend(*scatter.legend_elements(), title="Clusters")
#     plt.figtext(0.52, 0.9, 'Weak correlations in superfluid phase (projections appear to be randomly distributed) \n Strong correlations in Mott insulator phase (projections appear to be aligned)', ha='center', fontsize=10, color='red')
#     return scatter,

# ani_pca = animation.FuncAnimation(fig, update_pca, frames=len(pca_matrices), blit=False, repeat=False)

# # Save the PCA animation
# ani_pca.save(os.path.join(output_dir, 'pca_animation.gif'), writer='pillow', fps=1.5)

# plt.show()



# ################## SPDM ##################


# # Load the SPDM matrices
# spdm_matrices = []
# with open('spdm_matrices.csv', 'r') as file:
#     lines = file.readlines()
#     current_matrix = []
#     for line in lines:
#         if line.startswith("Matrix"):
#             if current_matrix:
#                 spdm_matrices.append(np.array(current_matrix))
#                 current_matrix = []
#         elif line.strip():  # Check if the line is not empty
#             current_matrix.append([float(x) for x in line.strip().split(',')])
#     if current_matrix:
#         spdm_matrices.append(np.array(current_matrix))

# # Create the animation for SPDM matrices
# fig, ax = plt.subplots(figsize=(10, 8))
# cax = ax.matshow(spdm_matrices[0], cmap='viridis')
# fig.colorbar(cax)

# def update_spdm(frame):
#     ax.clear()
#     matrix = spdm_matrices[frame]
#     cax = ax.matshow(matrix, cmap='viridis')
#     ax.set_title(f'SPDM Matrix {frame}')
#     return cax,

# ani_spdm = animation.FuncAnimation(fig, update_spdm, frames=len(spdm_matrices), blit=False, repeat=False)

# # Save the SPDM animation
# ani_spdm.save(os.path.join(output_dir, 'spdm_matrices.gif'), writer='pillow', fps=1.5)

# plt.show()