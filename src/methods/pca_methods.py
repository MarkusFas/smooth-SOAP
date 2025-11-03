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
from vesin import NeighborList
from memory_profiler import profile

from src.transformations.PCAtransform import PCA_obj
from src.methods.BaseMethod import FullMethodBase


class PCA(FullMethodBase):

    def __init__(self, descriptor, interval, root):
        self.name = 'PCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, method=self.name)
        

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
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples ) if label[2] == atom_type] for atom_type in self.descriptor.centers]

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) 
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        self.cov_tot = cov
        return mu, cov, [np.eye(c.shape[0]) for c in cov]


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

class PCAtest(FullMethodBase):

    def __init__(self, descriptor, interval, root):
        self.name = 'PCAtest'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, method=self.name)
        
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
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples ) if label[2] == atom_type] for atom_type in self.descriptor.centers]

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        kernel = delta
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) 
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
            buffer[:,fidx%self.interval,:] = new_soap_values

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        return mu, cov, [np.eye(c.shape[0]) for c in cov]


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov, cov_mu in zip(self.mean_cov_t, self.cov_mu_t)])
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


class PCAfull(FullMethodBase):

    def __init__(self, descriptor, interval, root):
        self.name = 'PCAfull'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, method=self.name)
        

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
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples ) if label[2] == atom_type] for atom_type in self.descriptor.centers]
    
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1))#gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= np.sum(kernel)
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        import matplotlib.pyplot as plt
        #plt.plot(kernel)
        #plt.savefig(self.label + '_kernel.png')
        #plt.close()
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) 
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #fig, ax = plt.subplots(1,1, figsize=(5,4))
                #ax.plot(buffer[0,:,0])
                #ax2 = ax.twinx()
                #ax2.plot(roll_kernel)
                #ax.scatter([fidx%self.interval + self.interval//2],[avg_soap[0,0]], color='red', s=50)
                #plt.savefig(self.label + f'_buffer_{fidx}.png')
                #plt.close()
               
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    mu_t = avg_soap[atom_type].mean(axis=0)
                    scatter_mut[atom_type_idx] += np.einsum(
                        "a,b->ab",
                        mu_t,
                        mu_t,
                    )

                    sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values
        #exit()
        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel) 
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        total_cov = mean_cov_t + cov_mu_t
        return mean_mu_t, total_cov, [np.eye(cov.shape[0]) for cov in total_cov]
        #return mean_mu_t, mean_cov_t, cov_mu_t

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov, cov_mu in zip(self.mean_cov_t, self.cov_mu_t)])
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
            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            ) 


class SpatialPCA(FullMethodBase):

    def __init__(self, descriptor, interval, sigma, root):
        self.name = 'SpatialPCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=sigma, method=self.name)


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
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples ) if label[2] == atom_type] for atom_type in self.descriptor.centers]
    
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_intra_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) 
            avg_soap_values = self.spatial_average_with_nl([system], new_soap_values)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    
                    #COV_intra_cluster, COV_inter_cluster = self.full_spatial_averaging(system, avg_soap, self.sigma)
                    mu_t = avg_soap[atom_type].mean(axis=0)
                    scatter_mut[atom_type_idx] += np.einsum(
                        "a,b->ab",
                        mu_t,
                        mu_t,
                    )

                    sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

                    #cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
                    cov_t[atom_type_idx] += COV_inter_cluster
                    cov_intra_t[atom_type_idx] += COV_intra_cluster
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = avg_soap_values

        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_cov_intra_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element): 
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            mean_cov_intra_t[atom_type_idx] = cov_intra_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel) 
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.mean_cov_intra_t = mean_cov_intra_t
        self.cov_mu_t = cov_mu_t

        # inter vs intra
        #return mean_mu_t, mean_cov_t, mean_cov_intra_t  #+ cov_mu_t
        # intra vs inter
        return mean_mu_t, mean_cov_intra_t, mean_cov_t  #+ cov_mu_t

        # FULL
        #return mean_mu_t, mean_cov_t + mean_cov_intra_t + cov_mu_t, [np.eye(c.shape[0]) for c in cov_mu_t]







    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.


        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov_intra), np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov_intra, mean_cov, cov_mu in zip(self.mean_cov_intra_t, self.mean_cov_t, self.cov_mu_t)])
        header = ["spatialCov_intra", "spatialCov_inter", "tempCov"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )


class SpatialPCA(FullMethodBase):

    def __init__(self, descriptor, interval, sigma, cutoff, root):
        self.name = 'SpatialPCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=sigma, method=self.name)
        self.descriptor_spatial = descriptor
        self.spatial_cutoff = cutoff
        self.label = self.label + f'cut_{self.spatial_cutoff}'
        
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
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples ) if label[2] == atom_type] for atom_type in self.descriptor.centers]

        #initialize neighbor list for spatial avg
        self.make_neighborlist(self.spatial_cutoff)

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_intra_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        logger = np.zeros((len(self.atomsel_element), len(systems), first_soap.shape[1],))
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) 
            avg_soap_values = self.spatial_average_with_nl(system, new_soap_values)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                avg_soap = self.spatial_averaging(system, avg_soap, self.sigma)
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    mu_t = avg_soap[atom_type].mean(axis=0)
                    scatter_mut[atom_type_idx] += np.einsum(
                        "a,b->ab",
                        mu_t,
                        mu_t,
                    )

                    sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms
                    logger[atom_type_idx][fidx] = mu_t
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before)
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = avg_soap_values
        np.savetxt(self.label + '_mu_t.csv', logger[0])
        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_cov_intra_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element): 
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel) 
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        # temp vs inter
        return mean_mu_t, cov_mu_t + mean_cov_t, [np.eye(c.shape[0]) for c in cov_mu_t] #mean_cov_t  #+ cov_mu_t



    def make_neighborlist(self, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, full_list=False)



    def spatial_average_with_nl(self, system, features):
        positions = system.positions  # (N,3) numpy array
        box = system.cell             # 3x3 array or None for non-periodic
        i, j, d, D = self.nlist.compute(
            points=positions,
            box=box,
            periodic=True,
            quantities="ijdD"
        )

        # filter pairs to only the centers you care about:
        mask = np.isin(i, self.selected_atoms)
        atom_types = system.types.numpy()
        i_sel, j_sel, d_sel, D_sel = i[mask], j[mask], d[mask], D[mask]
        averaged_features = np.zeros_like(features)
        self_weight = 1.0

        for idx, atom_idx in enumerate(self.selected_atoms):
            mask_i = i_sel == atom_idx 
            j_neighbors = j_sel[mask_i]
            d_neighbors = d_sel[mask_i]
            D_neighbors = D_sel[mask_i]

            # --- Deduplicate: keep only closest image per base atom
            if len(j_neighbors) > 0:
                unique_js = np.unique(j_neighbors)
                best_idx = np.array([
                    np.argmin(d_neighbors[j_neighbors == j]) 
                    for j in unique_js
                ])
                j_neighbors = unique_js
                d_neighbors = d_neighbors[best_idx]
                D_neighbors = D_neighbors[best_idx]

            mask_j = atom_types[j_neighbors] == 8
            if len(j_neighbors) == 0:
                averaged_features[idx] = features[idx]
                continue

            w = np.exp(-d_neighbors[mask_j]**2 / (2*self.sigma**2))
            sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            )
            feats = self.descriptor_spatial.calculate([system], sel_samples) 
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[idx]) / (self_weight + w.sum())
            averaged_features[idx] = h_i

        return averaged_features







    """
        self.nlist.update(system.positions)
        averaged_features = {}
        self_weight = 1.0
        for i in self.selected_atoms:
            j, d = self.nlist.get_neighbors(i, return_distances=True)
            if len(j) == 0:
                averaged_features[i] = features[i]
                continue

            w = np.exp(-d**2 / (2*self.sigma**2))
            self.descriptor_spatial.set_samples(j)
            feats = self.descriptor_spatial.calculate([system]) 
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[i]) / (self_weight + w.sum())
            averaged_features[i] = h_i

        return averaged_features

    """


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.


        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for  mean_cov, cov_mu in zip (self.mean_cov_t, self.cov_mu_t)])
        header = ["spatialCov_inter", "tempCov"]

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
            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )




class SpatialTempPCA(FullMethodBase):

    def __init__(self, descriptor, interval, sigma,cutoff, root):
        self.name = 'SpatialTempPCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=sigma, method=self.name)
        self.descriptor_spatial = descriptor
        self.spatial_cutoff = cutoff
        self.label = self.label + f'cut_{self.spatial_cutoff}'
    
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
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples ) if label[2] == atom_type] for atom_type in self.descriptor.centers]

        #initialize neighbor list for spatial avg
        self.make_neighborlist(self.spatial_cutoff)

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_intra_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        logger = np.zeros((len(self.atomsel_element), len(systems), first_soap.shape[1],))
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) 
            avg_soap_values = self.spatial_average_with_nl(system, new_soap_values)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                avg_soap = self.spatial_averaging(system, avg_soap, self.sigma)
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    mu_t = avg_soap[atom_type].mean(axis=0)
                    scatter_mut[atom_type_idx] += np.einsum(
                        "a,b->ab",
                        mu_t,
                        mu_t,
                    )

                    sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms
                    logger[atom_type_idx][fidx] = mu_t
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before)
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = avg_soap_values
        np.savetxt(self.label + '_mu_t.csv', logger[0])
        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_cov_intra_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element): 
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel) 
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        # temp vs inter
        return mean_mu_t, cov_mu_t, mean_cov_t #[np.eye(c.shape[0]) for c in cov_mu_t] #mean_cov_t  #+ cov_mu_t



    def make_neighborlist(self, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, full_list=False)


    def spatial_average_with_nl(self, system, features):
        positions = system.positions  # (N,3) numpy array
        box = system.cell             # 3x3 array or None for non-periodic
        i, j, d, D = self.nlist.compute(
            points=positions,
            box=box,
            periodic=True,
            quantities="ijdD"
        )
        # filter pairs to only the centers you care about:
        mask = np.isin(i, self.selected_atoms)
        atom_types = system.types.numpy()
        i_sel, j_sel, d_sel, D_sel = i[mask], j[mask], d[mask], D[mask]
        averaged_features = np.zeros_like(features)
        self_weight = 1.0
    


        for idx, atom_idx in enumerate(self.selected_atoms):
            mask_i = i_sel == atom_idx 
            j_neighbors = j_sel[mask_i]
            d_neighbors = d_sel[mask_i]
            mask_j = atom_types[j_neighbors] == 8
            if len(j_neighbors) == 0:
                averaged_features[idx] = features[idx]
                continue

            w = np.exp(-d_neighbors[mask_j]**2 / (2*self.sigma**2))
            sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            )
            feats = self.descriptor_spatial.calculate([system], sel_samples) 
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[idx]) / (self_weight + w.sum())
            averaged_features[idx] = h_i

        return averaged_features

    def spatial_average_with_nl(self, system, features):
        positions = system.positions  # (N,3) numpy array
        box = system.cell             # 3x3 array or None for non-periodic
        i, j, d, D = self.nlist.compute(
            points=positions,
            box=box,
            periodic=True,
            quantities="ijdD"
        )

        # filter pairs to only the centers you care about:
        mask = np.isin(i, self.selected_atoms)
        atom_types = system.types.numpy()
        i_sel, j_sel, d_sel, D_sel = i[mask], j[mask], d[mask], D[mask]
        averaged_features = np.zeros_like(features)
        self_weight = 1.0

        for idx, atom_idx in enumerate(self.selected_atoms):
            mask_i = i_sel == atom_idx 
            j_neighbors = j_sel[mask_i]
            d_neighbors = d_sel[mask_i]
            D_neighbors = D_sel[mask_i]

            # --- Deduplicate: keep only closest image per base atom
            if len(j_neighbors) > 0:
                unique_js = np.unique(j_neighbors)
                best_idx = np.array([
                    np.argmin(d_neighbors[j_neighbors == j]) 
                    for j in unique_js
                ])
                j_neighbors = unique_js
                d_neighbors = d_neighbors[best_idx]
                D_neighbors = D_neighbors[best_idx]

            mask_j = atom_types[j_neighbors] == 8
            if len(j_neighbors) == 0:
                averaged_features[idx] = features[idx]
                continue

            w = np.exp(-d_neighbors[mask_j]**2 / (2*self.sigma**2))
            sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            )
            feats = self.descriptor_spatial.calculate([system], sel_samples) 
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[idx]) / (self_weight + w.sum())
            averaged_features[idx] = h_i

        return averaged_features





    """
        self.nlist.update(system.positions)
        averaged_features = {}
        self_weight = 1.0
        for i in self.selected_atoms:
            j, d = self.nlist.get_neighbors(i, return_distances=True)
            if len(j) == 0:
                averaged_features[i] = features[i]
                continue

            w = np.exp(-d**2 / (2*self.sigma**2))
            self.descriptor_spatial.set_samples(j)
            feats = self.descriptor_spatial.calculate([system]) 
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[i]) / (self_weight + w.sum())
            averaged_features[i] = h_i

        return averaged_features

    """


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.


        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for  mean_cov, cov_mu in zip (self.mean_cov_t, self.cov_mu_t)])
        header = ["spatialCov_inter", "tempCov"]

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
            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )

 