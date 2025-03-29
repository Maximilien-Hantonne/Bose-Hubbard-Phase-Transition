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
save_path = os.path.join(current_dir, "figures")

# Science plots style
plt.style.use('science')

# Read the data from the file
with open(os.path.join(current_dir,"mean_field_pres.txt"), "r") as file:
    data = np.loadtxt(file)


#Extract mu, J, and Psi the superfluid density 
mus = data[:,0]
Js = data[:,1]
Psis = data[:,2]
Psis = np.where(Psis < 0, 0, Psis)

mus_unique = np.unique(mus)
Js_unique = np.unique(Js)
Js_grid, mus_grid = np.meshgrid(Js_unique, mus_unique)

fig = plt.figure(figsize=(11, 11))
contour = plt.contourf(Js_grid, mus_grid, Psis.reshape(len(mus_unique), len(Js_unique)), cmap='viridis')
cbar = plt.colorbar(contour, label=r'\ \ $\Psi$')
cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick label size
cbar.set_label(r'\ \ $\Psi$', fontsize=35, rotation = 0)  # Increase colorbar label size

plt.xlabel(r'\ $\frac{qJ}{U}$', fontsize=35)  # Increase x-axis label size
plt.ylabel(r'$\frac{\mu}{U}$\ ', fontsize=35, rotation = 0)  # Increase y-axis label size
# plt.ylim(0, 2.5)
plt.tick_params(axis='both', which='major', labelsize=25)  # Increase tick label size
plt.xlim(0,0.25)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "mean_field_pres.svg"))
plt.close(fig)


def fitp(J, n0):  
    return n0 - 0.5 - 0.5*J + 0.5*np.sqrt(1-2*(1+2*n0)*J + J**2)
def fitm(J, n0): 
    return n0 - 0.5 - 0.5*J - 0.5*np.sqrt(1-2*(1+2*n0)*J + J**2)

def Jmax(n0): 
    return (1+2*n0)*(1- np.sqrt(1-1/((1+2*n0)**2)))

fig2 = plt.figure(figsize = (11,11))
contour2 = plt.contourf(Js_grid, mus_grid, Psis.reshape(len(mus_unique), len(Js_unique)), cmap='Greys')
cbar2 = plt.colorbar(contour2, label=r'\ \ $\Psi$')
cbar2.ax.tick_params(labelsize=20)  # Increase colorbar tick label size
cbar2.set_label(r'\ \ $\Psi$', fontsize=35, rotation = 0)  # Increase colorbar label size

plt.xlabel(r'\ $\frac{qJ}{U}$', fontsize=35)  # Increase x-axis label size
plt.ylabel(r'$\frac{\mu}{U}$\ ', fontsize=35, rotation = 0)  # Increase y-axis label size
plt.ylim(0, 4)
plt.tick_params(axis='both', which='major', labelsize=25)  # Increase tick label size
plt.xlim(0,0.25)

plt.plot(Js[Js<Jmax(1)], fitm(Js[Js<Jmax(1)],1), label = "$n_0 = 1$", color = 'red', marker = '.', linestyle = 'None')
plt.plot(Js[Js<Jmax(1)], fitp(Js[Js<Jmax(1)],1), color = 'red',  marker = '.', linestyle = 'None')

plt.plot(Js[Js<Jmax(2)], fitm(Js[Js<Jmax(2)],2), label = "$n_0 = 2$", color = 'blue',  marker = '.', linestyle = 'None')
plt.plot(Js[Js<Jmax(2)], fitp(Js[Js<Jmax(2)],2), color = 'blue',  marker = '.', linestyle = 'None')

plt.plot(Js[Js<Jmax(3)], fitm(Js[Js<Jmax(3)],3), label = "$n_0 = 3$", color = 'green',  marker = '.', linestyle = 'None')
plt.plot(Js[Js<Jmax(3)], fitp(Js[Js<Jmax(3)],3), color = 'green',  marker = '.', linestyle = 'None')

plt.plot(Js[Js<Jmax(4)], fitm(Js[Js<Jmax(4)],4), label = "$n_0 = 4$", color = 'orange',  marker = '.', linestyle = 'None')
plt.plot(Js[Js<Jmax(4)], fitp(Js[Js<Jmax(4)],4), color = 'orange',  marker = '.', linestyle = 'None')
plt.legend(fontsize = 35, loc = 'center right')

plt.tight_layout()
plt.savefig(os.path.join(save_path, "mean_field_pres_grey.png"))
plt.close(fig2)