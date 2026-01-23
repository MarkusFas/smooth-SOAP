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
import vesin
from memory_profiler import profile



class PETMAD_descriptor():
    """
    This function has been replaced with model_soap 
    """
    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, selected_atoms=None):
        
        self.centers = centers
        self.neighbors = neighbors
        self.id = f"PETMAD"
        self.petmad = load_atomistic_model('data/PET-MAD.pt')
        self.nl_options = self.petmad.requested_neighbor_lists()[0]
        self.selected_samples = None
        #TODO default to all atoms in the trajectory
        self.output = ModelOutput(
            quantity='features', # mtt::aux::energy_last_layer_features
            unit='',
            per_atom=True,
            explicit_gradients=[],
        )

    
    def set_samples(self, selected_atoms):
        self.selected_samples = Labels(
            names=["system", "atom"],
            values=torch.tensor([[0, i] for i in selected_atoms], dtype=torch.int64),
        )
        
        self.options = ModelEvaluationOptions(
            length_unit='angstrom', 
            outputs={
                'mtt::aux::energy_last_layer_features': self.output, 
            }, # check features, check 'mtt::aux::energy_last_layer_features'
            selected_atoms=self.selected_samples,
        )

    def calculate(self, systems, selected_samples=None):

        if selected_samples is None:
            selected_samples = self.selected_samples
        self.options = ModelEvaluationOptions(
            length_unit='angstrom', 
            outputs={
                'mtt::aux::energy_last_layer_features': self.output, 
            }, # check features, check 'mtt::aux::energy_last_layer_features'
            selected_atoms=selected_samples,
        )
        #systems = systems_to_torch(structures, dtype=torch.float32)
        systems_new = []
        for i, system in enumerate(systems):
            system = system.to(torch.float32)
            #atoms = structures[i]
            nlist = vesin.NeighborList(cutoff=4.5, full_list=True)
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
            ).to(torch.float32)
            system.add_neighbor_list(self.nl_options, neighbors)
            systems_new.append(system.to(torch.float32))

        
        model = self.petmad(systems_new, 
            options=self.options,
            check_consistency=True,
        )
        self.soap_block = model['mtt::aux::energy_last_layer_features'].block()
        features = self.soap_block.values.numpy()
        return features




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
            if n_cumulants > 4:
                c[4] = mu[4] - 10 * mu[1] * mu[2]
            # Broadcast cumulant values to N samples
            cumulant_matrix.append(np.tile(c, (N, 1)))
        
        # Concatenate all cumulant blocks for each feature
        X_cumulant = np.hstack(cumulant_matrix)
        return X_cumulant



"""
in case sel atoms still doesnt work
new_map = mts.split(
            features,
            axis="samples",
            selections=[
                atomsel,
            ],
        )
    #feat = mean_over_samples(new_map[0], sample_names=["atom"]) 
    
    return new_map[0].block().values.numpy()
"""