import matplotlib.pyplot as plt
import numpy as np
import scienceplots 
import os

# Enable LaTex rendering in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, "figures")

# Science plots style
# plt.style.use(['science', 'grid'])

# Read the data from the file
with open("mean-field.txt", "r") as file:
    psi0 = np.loadtxt(file, max_rows=1)
    data = np.loadtxt(file, skiprows=1)


#Extract mu, J, and Psi the superfluid density 
mus = data[:,0]
Js = data[:,1]
Psis = data[:,2]

mus_unique = np.unique(mus)
Js_unique = np.unique(Js)
mus_grid, Js_grid = np.meshgrid(mus_unique, Js_unique)

fig = plt.figure(figsize=(18, 6))
contour = plt.contourf(Js_grid, mus_grid, Psis.reshape(len(Js_unique), len(mus_unique)), cmap='viridis')
plt.colorbar(contour, label=r'$\Psi$')
plt.xlabel(r'$\frac{qJ}{U}')
plt.ylabel(r'$\frac{\mu}{U}')
plt.text(0.5, 0.9, r'Ansatz: $\Psi0 = {psi0}$'.format(psi0=psi0), fontsize=10, transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "mean_field_test1.png"))
plt.close(fig)

