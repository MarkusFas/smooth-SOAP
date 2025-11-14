import sys

from ase.io import read
import numpy as np

plumed_fname = 'plumed.dat'
selected_type = 8          # example, use the actual type for O in your system
zmin, zmax = 10.0, 20.0

# Load the LAMMPS data file
atoms = read(sys.argv[1], format="lammps-data")  # ASE Atoms object

positions = atoms.get_positions()
types = atoms.get_atomic_numbers()

# Find all unique atom types
unique_types = np.unique(types)

# Collect indices by type (1-based for PLUMED)
atom_indices_by_type = {t: np.where(types == t)[0] + 1 for t in unique_types}

selected_atoms = [
    i + 1 for i, (t, pos) in enumerate(zip(types, positions))
    if t == selected_type and zmin <= pos[2] <= zmax
]


# ---------------------------
# Write plumed.dat
# ---------------------------
with open(plumed_fname, "w") as f:
    f.write("# Automatically generated plumed.dat\n\n")

    # Example SOAP block
    f.write(
        "soap_selected: METATOMIC ...\n"
        "      MODEL=soap_cv.pt\n"
        "      EXTENSIONS_DIRECTORY=extensions\n"
    )

    species_mapping = {t: i + 1 for i, t in enumerate(unique_types)}
    # Assign SPECIES dynamically
    for t in unique_types:
        f.write(f"      SPECIES{species_mapping[t]}={','.join(str(idx) for idx in atom_indices_by_type[t])}\n")
    f.write(f"      SPECIES_TO_TYPES={','.join(str(t) for t in unique_types)}\n")
    if selected_atoms:
        f.write(f"      SELECTED_ATOMS={','.join(str(idx) for idx in selected_atoms)}\n")

    # Printing directives
    f.write(
        "PRINT ARG=soap_selected FILE=soap_selected_data STRIDE=1 FMT=%8.4f\n"
        "CV: SUM ARG=soap_selected PERIODIC=NO\n"
        "PRINT ARG=CV FILE=COLVAR STRIDE=10\n"
        "FLUSH STRIDE=10\n"
    )

print(f"plumed.dat written with {len(selected_atoms)} selected atoms")