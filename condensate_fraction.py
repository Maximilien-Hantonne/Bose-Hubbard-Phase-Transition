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

# Save the plot
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, "figures")
plt.savefig(os.path.join(save_path, 'condensate_fractions.svg'))


with open('build/phase5.txt', 'r') as file:
    fixed_param_line = file.readline().strip().split()
    fixed_param = fixed_param_line[0]
    fixed_value = float(fixed_param_line[1])
    data = np.loadtxt(file)
    x_values = data[:100, 1] / fixed_value
    fc5 = data[:100,5]

# with open('buile/phase5.txt', 'r') as file:
#     fixed_param_line = file.readline().strip().split()
#     fixed_param = fixed_param_line[0]
#     fixed_value = float(fixed_param_line[1])
#     data5 = np.loadtxt(file)



# Plot the condensate fraction 
fig = plt.figure(figsize=(10, 7))
plt.scatter(x_values,fc5, color = 'blue')
plt.xlabel("Normalized Interaction strength ($U/J$)", fontsize = 35)
plt.ylabel("Condensate fraction $f_c$", fontsize = 35) 
plt.tick_params(axis='both', which='major', labelsize=25)  # Increase tick label size

plt.savefig(os.path.join(save_path, 'condensate_fraction.svg'))
plt.tight_layout()
plt.close(fig)