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

# Plot the condensate fraction 
fig2 = plt.figure(figsize=(10, 7))