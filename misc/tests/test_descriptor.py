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
)
from featomic.torch import SoapPowerSpectrum


class CumulantSOAP_CV(torch.nn.Module):
    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, projection_matrix=None):
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
        test = torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=k], dtype=torch.int32)
        self.selected_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=k], dtype=torch.int32),
        )

        self.id = f"SOAP_{cutoff}{max_angular}{max_radial}_{centers}"
        
        if projection_matrix is not None:
            self.register_buffer("projection_matrix", torch.tensor(projection_matrix.copy()).T)#[0].T)
        else:
            self.projection_matrix=None

    def calculate(self, systems, selected_samples=None):
        if selected_samples is None:
            selected_samples = self.selected_samples

        soap = self.calculator(
            systems,
            selected_samples=selected_samples,
            selected_keys=self.selected_keys,
        )

        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        self.soap_block = soap.block()
        return self.soap_block.values.numpy()

    def compute_cumulants_fwd(self, X: torch.Tensor, n_cumulants: int) -> torch.Tensor:
        """
        TorchScript-friendly computation of cumulants.

        X: (N, P) tensor
        n_cumulants: number of cumulants per feature
        returns: (N, P * n_cumulants) tensor
        """
        # ensure float
        X = X.float()
        N, P = X.shape  # Python ints

        # Preallocate output
        out = torch.empty((1, P * n_cumulants), dtype=X.dtype, device=X.device)

        # Temporary tensors reused per feature
        moments = torch.empty((n_cumulants,), dtype=X.dtype, device=X.device)
        c = torch.empty((n_cumulants,), dtype=X.dtype, device=X.device)

        jbase = 0
        for j in range(P):
            x = X[:, j]
            m = torch.mean(x)
            centered = x - m
            k = 1
            while k <= n_cumulants:
                moments[k - 1] = torch.mean(centered ** k)
                k += 1
            # 1st cumulant
            c[0] = m
            # 2nd cumulant
            if n_cumulants > 1:
                c[1] = moments[1 - 1] 
            # 3rd cumulant
            if n_cumulants > 2:
                c[2] = moments[2 - 1] 
            # 4th cumulant
            if n_cumulants > 3:
                mu2 = moments[1]
                mu4 = moments[3 - 1]
                c[3] = mu4 - 3.0 * (mu2 * mu2)
            # 5th cumulant
            if n_cumulants > 4:
                mu2 = moments[1]
                mu3 = moments[2]
                mu5 = moments[5 - 1]
                c[4] = mu5 - 10.0 * mu2 * mu3

            c_row = c.unsqueeze(0)  # .expand(N, n_cumulants)

            out[:, jbase:jbase + n_cumulants] = c_row
            jbase += n_cumulants
        return out


    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        
        if "features" not in outputs:
            return {}

        if outputs["features"].per_atom:
            raise ValueError("per_atom=True is not supported")

        if len(systems[0]) == 0:
            # PLUMED is trying to determine the size of the output
            projected = torch.zeros((0,1), dtype=torch.float64)
            samples = Labels(["system", "atom"], torch.zeros((0, 2), dtype=torch.int32))
            properties = Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1))
            
        else:
            soap = self.calculator(systems, selected_samples=selected_atoms, selected_keys=self.selected_keys)
            soap = soap.keys_to_samples("center_type")
            soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])#self.neighbor_type_pairs)

            soap_block = soap.block()
            
            X = self.compute_cumulants_fwd(soap_block.values, 3) # ATTENTION HARD CODED 3 
            projected = torch.einsum('ij,jk->ik',(X - self.mu), self.projection_matrix[:,self.proj_dims])#, dtype=torch.float64)

            #samples = soap_block.samples.remove("center_type")
            samples = Labels(["system", "atom"], torch.zeros((1, 2), dtype=torch.int32))
            properties = Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1))
            #print(self.proj_dims)

        block = TensorBlock(
            values=projected,
            samples=samples,
            components=[],
            #properties=Labels("soap_pca", torch.tensor([[0]])),
            #properties=Labels("soap_pca", torch.tensor([[0], [1]])),
            properties=properties,
        )
        cv = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block],
        )
        return {"features": cv}#, "soaps": soap }

    def set_projection_dims(self, dims):
        self.proj_dims = dims

    def set_projection_mu(self, mu):
        self.mu = torch.tensor(mu, dtype=torch.float64)

    def set_projection_matrix(self,matrix):
        self.projection_matrix=torch.tensor(matrix.copy())

    def save_model(self, path='.', name='soap_model'):
        capabilities = ModelCapabilities(
            outputs={"features": ModelOutput(per_atom=False)},
            interaction_range=10.0,
            supported_devices=["cpu"],
            length_unit="A",
            atomic_types=self.neighbors,
            dtype="float64",
        )
        
        metadata = ModelMetadata(name="Collective Variable test")
        self.eval()
        model = AtomisticModel(self, metadata, capabilities)
        model.save("{}/{}.pt".format(path,name), collect_extensions=f"{path}/extensions")

dims = 4
N = 576
mu = np.zeros(576)
matrix = np.eye(576)

descriptor = CumulantSOAP_CV(5, 3, 3, [8], [8,1])
descriptor.set_projection_dims(dims)
descriptor.set_projection_mu(mu)
descriptor.set_projection_matrix(matrix)
descriptor.save_model(path='.', name='test_cv')
