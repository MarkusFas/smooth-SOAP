import torch 
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import moment
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    systems_to_torch,
    load_atomistic_model
)
from featomic.torch import SoapPowerSpectrum
import argparse

class SOAP_CV_distinct(torch.nn.Module):
    def __init__(self, model, zmin, zmax):
        super().__init__()
        self.model = model
        self.zmin = zmin
        self.zmax = zmax
        
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:

        if selected_atoms is None:
            eval_options = ModelEvaluationOptions(
                length_unit='',
                outputs=outputs,
                selected_atoms=selected_atoms,
            )
            return self.model(systems, eval_options, check_consistency=True)

        pos = systems[0].positions[selected_atoms.values[:,1]]
        mask = (pos[:, 2] > self.zmin) & (pos[:, 2] < self.zmax)
        #dtype = selected_atoms.values.dtype

        print('pos', pos.shape)
        print('selatoms', selected_atoms.values.shape)
        print('mask', mask.shape)
        selected_atoms_new = Labels(
            names=selected_atoms.names,
            values=selected_atoms.values[mask],
        )
        print('newselatoms', selected_atoms_new.values.shape)
        eval_options = ModelEvaluationOptions(
            length_unit='',
            outputs=outputs,
            selected_atoms=selected_atoms_new,
        )
        return self.model(systems, eval_options, check_consistency=True) #, "soaps": soap }


    def save_model(self, path='.', name='wrapper'):
        capabilities = self.model.capabilities()
        metadata = self.model.metadata()
        wrapper = AtomisticModel(self, metadata, capabilities)
        wrapper.save("{}/{}.pt".format(path, name), collect_extensions=f"{path}/extensions")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SOAP wrapper based on z-coordinate filtering')
    parser.add_argument('model_path', type=str, help='model to wrap')
    parser.add_argument('--zmin', type=float, default=-np.inf, help='minimum z-coordinate')
    parser.add_argument('--zmax', type=float, default=np.inf, help='maximum z-coordinate')

    model_path = parser.parse_args().model_path
    zmin = parser.parse_args().zmin
    zmax = parser.parse_args().zmax

    model = load_atomistic_model(model_path)
    wrapper = SOAP_CV_distinct(model, zmin, zmax)
    wrapper.eval()
    wrapper.save_model(path='.', name=f'soap_wrapper_zmin{zmin}_zmax{zmax}')

    import vesin
    from ase.io import read, write
    #structures = read('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/icemeltinterface/nobias/positions.lammpstrj', index=':')
    structures = read('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/icemeltinterface/nobias/short.lammpstrj', index=':')
    structures = structures[:10]
    systems = systems_to_torch(structures, dtype=torch.float64)

    systems_new = []
    for i, system in enumerate(systems):
        
        #atoms = structures[i]
        nlistoptions = wrapper.model.requested_neighbor_lists()[0]
        print(nlistoptions)
        nlist = vesin.NeighborList(cutoff=nlistoptions.cutoff, full_list=nlistoptions.full_list) 
        i, j, S, D = nlist.compute(
            points=system.positions,
            box=system.cell, 
            periodic=True,
            quantities="ijSD"
        )
        #i, j, S, D = ase_neighbor_list(quantities="ijSD", a=atoms, cutoff=4.5)
        i = torch.from_numpy(i.astype(int))
        j = torch.from_numpy(j.astype(int))
        neighbor_indices = torch.stack([i, j], dim=1)
        neighbor_shifts = torch.from_numpy(S.astype(int))

        sample_values = torch.hstack([neighbor_indices, neighbor_shifts])
        samples = Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=sample_values,
        )

        neighbors = TensorBlock(
            values=torch.from_numpy(D).reshape(-1, 3, 1),
            samples=samples,
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )
        system.add_neighbor_list(nlistoptions, neighbors)
        systems_new.append(system)

    systems = systems_new
    selected_atoms = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, j] for j in np.arange(0, len(structures[0]), 3)]))
    
    model_output = ModelOutput(per_atom=False)

    cv = wrapper(
        systems=systems,
        outputs={"features": model_output},
        selected_atoms=selected_atoms,
    )

    print(cv)