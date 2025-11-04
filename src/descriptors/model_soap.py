
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import LinearRegression
import torch 

from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    systems_to_torch,
)
from featomic.torch import SoapPowerSpectrum

class SOAP_CV(torch.nn.Module):
    def __init__(self, trj, cutoff, max_angular, max_radial, centers, neighbors, projection_matrix=None):
        super().__init__()
        HYPER_PARAMETERS = {
            "cutoff": {
                "radius": cutoff, #4 #5 #6
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
            },
            "density": {
                "type": "Gaussian",
                "width": 0.25, #changed from 0.3
            },
            "basis": {
                "type": "TensorProduct",
                "max_angular": max_angular, #8
                "radial": {"type": "Gto", "max_radial": max_radial}, #6
            },
        }
        self.calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
        self.centers = centers
        self.neighbors = neighbors
        self.selected_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
                k], dtype=torch.int32),
        )
        types=[i.number for j in trj for w in j for i in w]
        self.atomic_types= sorted(set(types), key=types.index) #[torch.tensor([i for i in centers]+[j for j in neighbors if j not in centers ], dtype=torch.int32)]
        self.id = f"SOAP_{cutoff}{max_angular}{max_radial}_{centers}"
        
        if projection_matrix !=None:
            self.register_buffer("projection_matrix", torch.tensor(trans_matrix.copy()).T)#[0].T)
        else:
            self.projection_matrix=None
    
    def set_samples(self, selected_atoms):
        self.selected_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
        )

    def set_projection_dims(self, dims):
        self.proj_dims = dims

    def calculate(self, systems):
        
        soap = self.calculator(
            systems,
            selected_samples=self.selected_samples,
            selected_keys=self.selected_keys,
        )
        
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        self.soap_block = soap.block()
        return self.soap_block.values.numpy() 

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "features" not in outputs:
            return {}

        if not outputs["features"].per_atom:
            raise ValueError("per_atom=False is not supported")

        if len(systems[0]) == 0:
            # PLUMED is trying to determine the size of the output
            projected = torch.zeros((0,1), dtype=torch.float64)
            samples = Labels(["system", "atom"], torch.zeros((0, 2), dtype=torch.int32))
        else:
            soap = self.calculator(systems, selected_samples=selected_atoms, selected_keys=self.selected_keys)
            soap = soap.keys_to_samples("center_type")
            soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])#self.neighbor_type_pairs)

            soap_block = soap.block()
            projected = soap_block.values @ self.projection_matrix[self.proj_dims].T
            projected=projected #.unsqueeze(1)

            samples = soap_block.samples.remove("center_type")

        block = TensorBlock(
            values=projected,
            samples=samples,
            components=[],
            #properties=Labels("soap_pca", torch.tensor([[0]])),
            #properties=Labels("soap_pca", torch.tensor([[0], [1]])),
            #properties=Labels("soap_pca", torch.tensor([[0], [1], [3]]).T),
            properties=Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1)),
        )
        cv = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block],
        )
        return {"features": cv}#, "soaps": soap }

    def set_projection_matrix(self,matrix):
        self.projection_matrix=torch.tensor(matrix.copy())

    def save_model(self, path='.', name='soap_model'):
        capabilities = ModelCapabilities(
            outputs={"features": ModelOutput(per_atom=True)},
            interaction_range=10.0,
            supported_devices=["cpu"],
            length_unit="A",
            atomic_types=self.neighbors,
            dtype="float64",
        )
        
        metadata = ModelMetadata(name="Collective Variable test")
        model = AtomisticModel(self, metadata, capabilities)
        model.save("{}/{}.pt".format(path,name), collect_extensions=f"{path}/extensions")

#    def fit_ridge( alpha=0.3):
#        #RIDGE
#        
#        alpha=0.3
#        clf = Ridge(alpha=alpha,fit_intercept = False)
#        #clf=LinearRegression(fit_intercept = False)
#        clf.fit(np.vstack([gfeatures[i] for i in ids_atoms]), struct_soap_pca[:,:])
#        self.save_transformation_matrix(clf.coef_)    
#    
#    def save_transformation_matrix(coeffs):
#        torch.save(torch.tensor(coeffs),'model_test.pt'.format(cutoff,amax,rmax))





