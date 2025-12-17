import ase.io
import sys 
from ase.data import chemical_symbols

atoms = ase.io.read(sys.argv[1])
mask = [atom.symbol != "W" for atom in atoms]

# Keep only atoms that are not W
atoms = atoms[mask]
print(chemical_symbols)
print(atoms)

ase.io.write(sys.argv[2], atoms, format='lammps-data')
