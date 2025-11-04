import torch 
from abc import ABC, abstractmethod
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

from src.transformations.PCAtransform import PCA_obj
from src.methods.BaseMethod import FullMethodBase


class TICA(FullMethodBase):

    def __init__(self, descriptor, interval, lag, sigma, root):
        self.name = 'TICA'
        super().__init__(descriptor, interval, lag=lag, root=root, sigma=sigma, method=self.name)

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
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_t = np.zeros((first_soap.shape[0], self.lag+1, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        corr_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_corr = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        nsmp_corr = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        ntimesteps_corr = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            buffer[:,fidx%self.interval,:] = new_soap_values
            
            if fidx >= self.interval-1:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                roll_kernel = np.roll(kernel, (fidx+1)%self.interval)
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #avg_soap = self.spatial_averaging(system, avg_soap, self.sigma)
                buffer_t[:,fidx%(self.lag+1),:] = avg_soap

            if fidx >= self.interval + self.lag - 1:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                soap_0 = buffer_t[:,fidx%(self.lag+1),:]
                soap_lag = buffer_t[:,(fidx+1)%(self.lag+1),:]

                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    # C0
                    sum_soaps[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                    sum_soaps[atom_type_idx] += soap_lag[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_0[atom_type], soap_0[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_lag[atom_type], soap_lag[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += 2*len(atom_type)

                    sum_soaps_corr[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                    corr_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_0[atom_type], soap_lag[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp_corr[atom_type_idx] += len(atom_type)
                    ntimesteps_corr[atom_type_idx] += 1

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mu_corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            
            mu_corr[atom_type_idx] = sum_soaps_corr[atom_type_idx]/nsmp_corr[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            corr[atom_type_idx] = corr_t[atom_type_idx]/nsmp_corr[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        self.cov = cov
        self.corr = corr
        self.mu = mu
        self.mu_corr = mu_corr
        if self.lag == 0:
            return mu, corr, [np.eye(m.shape[0]) for m in corr]
        return mu, corr, cov

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        Returns
        -------
        empty
        """
        metrics = np.array([self.mu[0], self.mu_corr[0], np.diag(self.cov[0]), np.diag(self.corr[0])])
        header = ["mu", "mu_corr", "cov", "corr"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )

        for trafo in self.transformations:
            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            ) 

#TODO : add lag to self.label
class TILDA(FullMethodBase):

    def __init__(self, descriptor, interval, lag, sigma, root):
        self.name = 'TILDA'
        super().__init__(descriptor, interval, lag=lag, root=root, sigma=sigma, method=self.name)

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
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_t = np.zeros((first_soap.shape[0], self.lag+1, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        corrA = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        corrB = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_A = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_B = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        nsmp_corr = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        ntimesteps_corr = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            buffer[:,fidx%self.interval,:] = new_soap_values
            
            if fidx >= self.interval-1:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                roll_kernel = np.roll(kernel, (fidx+1)%self.interval)
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #avg_soap = self.spatial_averaging(system, avg_soap, self.sigma)
                buffer_t[:,fidx%(self.lag+1),:] = avg_soap

            if fidx >= self.interval + self.lag - 1:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                soap_0 = buffer_t[:,fidx%(self.lag+1),:]
                soap_lag = buffer_t[:,(fidx+1)%(self.lag+1),:]

                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    # C0
                    sum_soaps[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                    sum_soaps[atom_type_idx] += soap_lag[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_0[atom_type], soap_0[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_lag[atom_type], soap_lag[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += 2*len(atom_type)

                    sum_soaps_A[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                    sum_soaps_B[atom_type_idx] += soap_lag[atom_type].sum(axis=0)
                    corrA[atom_type_idx] += np.einsum("ia,ib->ab", soap_0[atom_type], soap_0[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    corrB[atom_type_idx] += np.einsum("ia,ib->ab", soap_lag[atom_type], soap_lag[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp_corr[atom_type_idx] += len(atom_type)
                    ntimesteps_corr[atom_type_idx] += 1

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        covA = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        covB = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        muA = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        muB = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            #cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            
            #mu_corr[atom_type_idx] = sum_soaps_corr[atom_type_idx]/nsmp_corr[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            #corr[atom_type_idx] = corr_t[atom_type_idx]/nsmp_corr[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
            # muA 
            muA[atom_type_idx] = sum_soaps_A[atom_type_idx]/nsmp_corr[atom_type_idx]
           
            # muB
            muB[atom_type_idx] = sum_soaps_B[atom_type_idx]/nsmp_corr[atom_type_idx]

            # covA
            covA[atom_type_idx] = corrA[atom_type_idx]/nsmp_corr[atom_type_idx] - np.einsum('i,j->ij', muA[atom_type_idx], muA[atom_type_idx])
            
            # covB
            covB[atom_type_idx] = corrA[atom_type_idx]/nsmp_corr[atom_type_idx] - np.einsum('i,j->ij', muA[atom_type_idx], muA[atom_type_idx])
        
        # harmonic mean of covariances
        self.cov = np.array([np.linalg.inv(np.linalg.inv(0.5*(cov1 + cov1.T)) + np.linalg.inv(0.5*(cov2 + cov2.T))) for cov1, cov2 in zip(covA, covB)])
        self.corr = np.einsum('ij,ik->ijk', muA - muB, muA - muB)
        self.mu = 0.5 * (muA + muB)
        self.mu_corr = muA
        if self.lag == 0:
            return self.mu, self.corr, [np.eye(m.shape[0]) for m in corr]
        return self.mu, self.corr, self.cov

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        Returns
        -------
        empty
        """
        metrics = np.array([self.mu[0], np.diag(self.corr[0]), np.diag(self.cov[0])])
        header = ["mu", "corr", "cov"]

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


class IVAC(FullMethodBase):

    def __init__(self, descriptor, interval, max_lag, min_lag, lag_step, root):
        self.name = 'IVAC'
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.lag_step = lag_step
        super().__init__(descriptor, interval, lag='ivac', root=root, sigma=0, method=self.name)

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
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_t = np.zeros((first_soap.shape[0], self.max_lag+1, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        corr_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_corr = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        nsmp_corr = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        ntimesteps_corr = np.zeros(len(self.atomsel_element), dtype=int)
        #IVAC specific:
        lags = np.arange(self.min_lag, self.max_lag + self.lag_step, self.lag_step)
        delta_soap_lag = np.zeros((len(lags), first_soap.shape[0], first_soap.shape[1]))
        soap_0_mu = np.zeros((len(self.atomsel_element), first_soap.shape[1],))
        soap_lag_mu = np.zeros((len(self.atomsel_element), len(lags), first_soap.shape[1],))


        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            buffer[:,fidx%self.interval,:] = new_soap_values
            if fidx >= self.interval-1:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                roll_kernel = np.roll(kernel, (fidx+1)%self.interval)
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #avg_soap = self.spatial_averaging(system, avg_soap, self.sigma)
                buffer_t[:,fidx%(self.max_lag+1),:] = avg_soap

            for i, lag in enumerate(lags):
                if fidx >= self.interval + lag - 1:

                    # computes a contribution to the correlation function
                    # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                    soap_0 = buffer_t[:,fidx%(self.max_lag + 1),:]
                    soap_lag = buffer_t[:,(fidx-lag)%(self.max_lag + 1),:]
                    #soap_lags = [buffer_t[:,(fidx-lag)%(self.max_lag + 1),:] for lag in lags]
                    for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                        # C0
                        #delta_soap_0 = soap_0[atom_type] - soap_mu[atom_type_idx] 
                        sum_soaps[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                        cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_0[atom_type], soap_0[atom_type])# Ctau
                        #delta_soap_lag = soap_lag[atom_type] - soap_mu[atom_type_idx]
                        sum_soaps[atom_type_idx] += soap_lag[atom_type].sum(axis=0)
                        cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_lag[atom_type], soap_lag[atom_type])
                        #cov_t[atom_type_idx] += np.einsum("ia,ib->ab", delta_soap_lag, delta_soap_lag) #sum over all same atoms (have already summed over all times before)
                        #corr_t[atom_type_idx] += np.einsum("ia,ib->ab", delta_soap_lag, delta_soap_0)
                        corr_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_0[atom_type], soap_lag[atom_type])
                        #nsmp[atom_type_idx] += (len(soap_lags) + 1)*len(atom_type)
                        nsmp[atom_type_idx] += 2 * len(atom_type)
                        nsmp_corr[atom_type_idx] += len(atom_type)
                        #ntimesteps_corr[atom_type_idx] += 1
                        #sum_soaps_corr[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                        #delta_soap_0 = soap_0[atom_type] - soap_0_mu[atom_type_idx] 
                        
                        #soap_0_mu[atom_type_idx] += delta_soap_0.mean(axis=0) / ntimesteps_corr[atom_type_idx]
                        #TODO: test for only one
                        #for i, soap_lag in enumerate(soap_lags):
                            #delta_soap_lag[i] = soap_lag[atom_type] - soap_lag_mu[atom_type_idx, i]
                        #   delta_soap_lag[i] = soap_lag[atom_type] - soap_mu[atom_type_idx]
                            #soap_lag_mu[atom_type_idx, i] += delta_soap_lag[i].mean(axis=0) / ntimesteps_corr[atom_type_idx]
                            #sum over all same atoms (have already summed over all times before) 


        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        #mu_corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            
            #mu_corr[atom_type_idx] = sum_soaps_corr[atom_type_idx]/nsmp_corr[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            corr[atom_type_idx] = corr_t[atom_type_idx]/(nsmp_corr[atom_type_idx]) - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        self.cov = cov
        self.corr = corr
        self.mu = mu #soap_mu
        #self.mu_corr = mu_corr
        #return soap_lag_mu.mean(axis=1), corr, cov
        return mu, corr, cov
        #return soap_mu, corr, [np.eye(c.shape[0]) for c in corr]

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """

        for i, trafo in enumerate(self.transformations):
            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )