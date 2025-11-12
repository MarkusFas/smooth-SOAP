import torch 
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.stats import moment
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
    

    def compute_cumulants(self, X, n_cumulants):
        """
        Compute cumulants for each feature and concatenate them horizontally.
        
        Parameters
        ----------
        X : np.ndarray, shape (N, P)
            Data matrix with N samples and P features.
        n_cumulants : int
            Number of cumulants to compute per feature.
        
        Returns
        -------
        X_cumulant : np.ndarray, shape (N, P * n_cumulants)
            New feature matrix where cumulants of each original feature 
            are concatenated along the feature axis.
        """
        X = np.asarray(X)
        N, P = X.shape
        
        cumulant_matrix = []
        for j in range(P):
            x = X[:, j]
            m = np.mean(x)
            centered = x - m

            # Compute central moments up to n_cumulants
            mu = np.array([moment(centered, moment=i) for i in range(1, n_cumulants + 1)])
            c = np.zeros(n_cumulants)
            
            # First cumulants (mean, variance, skewness, kurtosis, ...)
            c[0] = m
            if n_cumulants > 1:
                c[1] = mu[1]                 # variance
            if n_cumulants > 2:
                c[2] = mu[2]                 # 3rd central moment
            if n_cumulants > 3:
                c[3] = mu[3] - 3 * mu[1]**2  # 4th cumulant (kurtosis-related)
            # higher orders could follow recursion, but are rarely stable
            
            # Broadcast cumulant values to N samples
            cumulant_matrix.append(np.tile(c, (N, 1)))
        
        # Concatenate all cumulant blocks for each feature
        X_cumulant = np.hstack(cumulant_matrix)
        return X_cumulant


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
    
