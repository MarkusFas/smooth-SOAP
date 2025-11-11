import torch 
import os
from abc import ABC, abstractmethod
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import ase.neighborlist
from itertools import chain
from vesin import NeighborList
from memory_profiler import profile
from sklearn.decomposition import PCA as skPCA
from src.transformations.PCAtransform import PCA_obj
from src.methods.BaseMethod import FullMethodBase
from sklearn.linear_model import Ridge 


class PCA_distinct(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'PCA_distinct'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)


    def predict(self, traj, selected_atoms):
        """
        Project new trajectory frames into the trained collective variable (CV) space.

        Parameters
        ----------
        traj : list[ase.Atoms]
            Trajectory to project.
        selected_atoms : list[int]
            Indices of atoms to project.

        Returns
        -------
        np.ndarray, shape (n_atoms, n_frames, n_components)
            Projected low-dimensional representation.
        """
        if self.transformations is None:
            raise RuntimeError("Call train() before predict().")

        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float64)

        projected_per_type = []

        for trafo in self.transformations:
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                descriptor_proj = trafo.project(descriptor)
                projected.append(descriptor_proj)
                # TODO:
                #self.ridge.fit(descriptor, descriptor_proj)
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)
    



    def compute_COV(self, traj):
        """
        Compute time-averaged SOAP covariance matrices for each atomic species.

        This method computes the temporal and ensemble covariance of SOAP 
        descriptors for different atomic species over a molecular dynamics 
        trajectory. It uses a Gaussian kernel to smooth SOAP vectors in time 
        and separates intra-atomic (within-atom) and inter-atomic (between-atoms)
        covariance contributions.

        Parameters
        ----------
        traj : ase.io.Trajectory or list of ase.Atoms
            Molecular dynamics trajectory containing atomic configurations 
            for which the SOAP descriptors are computed.

        Returns
        -------
        mean_mu_t : np.ndarray, shape (n_species, n_features)
            Time-averaged mean SOAP vector for each atomic species.
        mean_cov_t : np.ndarray, shape (n_species, n_features, n_features)
            Mean covariance of SOAP descriptors across all timesteps and atoms 
            of a given species.
        cov_mu_t : np.ndarray, shape (n_species, n_features, n_features)
            Temporal covariance of SOAP descriptor means (fluctuations in time).
        """
        systems = systems_to_torch(traj, dtype=torch.float64)
        soap_block = self.descriptor.calculate(systems[:1])
        first_soap = soap_block  
        # TODO: only implemented for one center type, could extend if necessary
        self.atomsel_atom = [[idx] for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == self.descriptor.centers[0]]
        #if soap_block.shape[0] == 1:
        #    self.atomsel_atom = [[0] for atom_type in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_atom), first_soap.shape[0], first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_atom),first_soap.shape[0], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_atom))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_atom), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_atom):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type]
                    cov_t[atom_type_idx] += np.einsum("ia,ib->iab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values

        mu = np.zeros((len(self.atomsel_atom), new_soap_values.shape[0], new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_atom), new_soap_values.shape[0], new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_atom):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        self.cov_tot = cov
        return mu[0], cov[0], [np.eye(c.shape[0]) for c in cov] #hard coded only for first center!
    


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(tot_cov)] for tot_cov in self.cov_tot])
        header = ["spatialCov", "tempCov"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )
        for i, trafo in enumerate(self.transformations):
            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            ) 

