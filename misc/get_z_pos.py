import ase.io
import matplotlib.pyplot as plt
import numpy as np
import chemiscope
from ase.visualize import view

sel_atoms = [51, 60, 63, 69, 72, 81, 84, 87, 90, 93, 147, 156, 159, 165, 168, 177, 180, 183, 186, 189, 243, 252, 255, 261, 264, 273, 276, 279, 282, 285, 339, 348, 351, 357, 360, 369, 372, 375, 378, 381]
trj = ase.io.read('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/metad/metad_interface/CumulantPCA/metaD_250_01/positions.lammpstrj', index=':')
z_pos = []
pos = trj[0].get_positions()
z_min = np.min(pos)
z_max = np.max(pos)
for atoms in trj[:1]:
    pos = atoms[sel_atoms].get_positions()
    z_pos.append(pos[:,2])

z_pos = np.array(z_pos)
plt.scatter(np.arange(len(sel_atoms)), z_pos[0,:])
plt.ylim(z_min, z_max)
plt.show()
view(atoms)
chemiscope.show(
    trj,
    mode='structure'
)
