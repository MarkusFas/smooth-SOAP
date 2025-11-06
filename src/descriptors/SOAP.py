import torch 
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import ase.neighborlist
from vesin import ase_neighbor_list
from memory_profiler import profile


class SOAP_descriptor():
    """
    This function has been replaced with model_soap 
    """
    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, selected_atoms=None):
        HYPER_PARAMETERS = {
            "cutoff": {
                "radius": cutoff, #4 #5 #6
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
            },
            "density": {
                "type": "Gaussian",
                "width": 0.3, #changed from 0.3
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
        self.sel_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
                k], dtype=torch.int32),
        )
        self.id = f"SOAP_{cutoff}{max_angular}{max_radial}_{centers}"
        
        self.sel_samples = None
        #TODO default to all atoms in the trajectory

        if selected_atoms is not None:
            self.sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
            )

    
    def set_samples(self, selected_atoms):
        self.sel_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
        )

    def calculate(self, systems, sel_samples=None):
        if sel_samples is None:
            sel_samples = self.sel_samples
        soap = self.calculator(
            systems,
            selected_samples=sel_samples,
            selected_keys=self.sel_keys,
        )
        
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        self.soap_block = soap.block()
        return self.soap_block.values.numpy()
    

class SOAP_descriptor_special():
    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, selected_atoms=None):
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
        self.sel_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
                k], dtype=torch.int32),
        )
        self.id = f"SOAP_{cutoff}{max_angular}{max_radial}_{centers}"
        
        self.sel_samples = None
        #TODO default to all atoms in the trajectory

        if selected_atoms is not None:
            self.sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
            )

    
    def set_samples(self, selected_atoms):
        self.sel_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
        )

    def calculate(self, systems, sel_samples=None):
        if sel_samples is None:
            sel_samples = self.sel_samples
        soap = self.calculator(
            systems,
            selected_samples=sel_samples,
            selected_keys=self.sel_keys,
        )
        
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        self.soap_block = soap.block()
        soap_block = self.soap_block.values.numpy()
        N,S = soap_block.shape
        return soap_block.reshape(1, -1) #TODO: return numpy
    
