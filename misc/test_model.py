
from ase.io import read, write
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    systems_to_torch,
)
from metatomic.torch import load_atomistic_model, check_atomistic_model
from ase.atoms import Atoms
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

def sort_frames(frames):
    sortedframes=[]
    for b in frames:
        a=Atoms([i for j in range(100) for i in b if i.number==j], cell=b.cell, pbc=[True, True, True])
        sortedframes.append(a)
    return sortedframes

ifile="/home/hanna/data/averagedSOAP_metadynamics/BaTiO3/mds/fs_dump_flipping2/positions_shorter30000.extxyz"

frames=read(ifile, '-5:')

frames=sort_frames(frames)
systems = systems_to_torch(frames, dtype=torch.float64)
#print(len(systems))

model=load_atomistic_model('model_soap.pt', extensions_directory='extensions')

selected_atoms=[202, 212, 204, 223, 242, 247, 211, 248, 235, 237, 192]

#print(torch.tensor([[0]*len(selected_atoms),selected_atoms], dtype=torch.int64).T)
selected_samples = Labels(
            names=["system","atom"],
            values=torch.tensor([[0]*len(selected_atoms),selected_atoms], dtype=torch.int64).T,
        )

#print(selected_samples)

opts = ModelEvaluationOptions(
    length_unit="A",
    outputs={"features": ModelOutput(quantity="", per_atom=True)},
    selected_atoms=selected_samples,
)


prediction = model.forward(systems, options=opts, check_consistency=False)

print(prediction["features"][0].properties)

print(prediction["features"][0].values)



