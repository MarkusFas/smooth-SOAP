import ase
from ase.io import read
from collections import Counter
atoms = read('data/chignolin/0_whole_centered.pdb')

types = atoms.get_chemical_symbols()
counts = Counter(types)

print(counts)