import torch
from abc import ABC, abstractmethod
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from vesin import NeighborList
from memory_profiler import profile
from sklearn.linear_model import Ridge

from src.transformations.PCAtransform import PCA_obj
from src.methods.BaseMethod import FullMethodBase


class SpatialIVAC(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, n_cumulants, root):
        self.name = 'SpatialIVAC'
        self.n_cumulants = n_cumulants
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
                cum_descriptor = self.descriptor.compute_cumulants(descriptor, self.n_cumulants)
                descriptor_proj = trafo.project(cum_descriptor)
                projected.append(descriptor_proj)
                # TODO:
                #self.ridge.fit(descriptor, descriptor_proj)
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)
    
    
    def predict_ridge(self, traj, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float64)
       
        projected_per_type = []

        for idx, trafo in enumerate(self.transformations):
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                cum_descriptor = self.descriptor.compute_cumulants(descriptor, self.n_cumulants)
            
                #ridge_pred = self.ridge[idx].predict(cum_descriptor)
                ridge_pred = self.ridge[idx].predict(descriptor)
                projected.append(ridge_pred)
               
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type
        


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
        first_soap = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        first_soap_cum = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        
        self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_cum = np.zeros((first_soap.shape[0], self.interval, first_soap_cum.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_t_cum = np.zeros((len(self.atomsel_element), first_soap_cum.shape[1], first_soap_cum.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_cum = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            cum_soap_values = self.descriptor.compute_cumulants(new_soap_values, self.n_cumulants)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                avg_soap_cum = np.einsum("j,ija->ia", roll_kernel, buffer_cum) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    
                    sum_soaps_cum[atom_type_idx] += avg_soap_cum[atom_type].sum(axis=0)
                    cov_t_cum[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap_cum[atom_type], avg_soap_cum[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values
            buffer_cum[:,fidx%self.interval,:] = cum_soap_values

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mu_cum = np.zeros((len(self.atomsel_element), cum_soap_values.shape[1]))
        cov_cum = np.zeros((len(self.atomsel_element), cum_soap_values.shape[1], cum_soap_values.shape[1]))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            mu_cum[atom_type_idx] = sum_soaps_cum[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov_cum[atom_type_idx] = cov_t_cum[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        self.cov_tot = cov
        return mu, cov_cum, cov


    def fit_ridge_nonincremental(self, traj):
        systems = systems_to_torch(traj, dtype=torch.float64)
        soap_block = self.descriptor.calculate(systems[:1], selected_samples=self.descriptor.selected_samples)
        print(soap_block.shape)
        first_soap = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum() #kernel = delta
        self.ridge = {}


        for idx, trafo in enumerate(self.transformations):
            self.ridge[idx] = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            avg_soap_proj = trafo.project(first_soap) 
            print('avg_soap_proj',avg_soap_proj.shape)
            soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
            avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
            for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                new_soap_values = self.descriptor.calculate([system], selected_samples=self.descriptor.selected_samples)
                cum_soap_values = self.descriptor.compute_cumulants(new_soap_values, self.n_cumulants)
                new_soap_values = None
                if fidx >= self.interval:
                    roll_kernel = np.roll(kernel, fidx%self.interval)
                    # computes a contribution to the correlation function
                    # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                    avg_soap_proj = trafo.project(avg_soap)
                    #print('projshape', avg_soap_proj.shape)
                    #print('nonprog.shape',new_soap_values.shape)
                    soap_values[:,fidx-self.interval,:] = cum_soap_values
                    avg_soaps_projs[:,fidx-self.interval,:] = avg_soap_proj
                buffer[:,fidx%self.interval,:] = cum_soap_values
            print('soapvals',soap_values.dtype)
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            #soap_values=soap_values.reshape((soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2]))
            #avg_soaps_projs=avg_soaps_projs.reshape((avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
            avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
            #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
            #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            self.ridge[idx].fit(soap_values, avg_soaps_projs)
      

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

            torch.save(
                torch.tensor(self.cov1[i]),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_cov1.pt",
            ) 

            torch.save(
                self.descriptor.soap_block.properties.values,
                self.label + f"_center{self.descriptor.centers[i]}" + f"_properties.pt",
            )



class SpatialIVACnorm(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, n_cumulants, root):
        self.name = 'SpatialIVAC'
        self.n_cumulants = n_cumulants
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)



    def make_neighborlist(self, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, full_list=True) # TODO: debatable if full lsit or not

    def spatial_correlate(self, system, sel_atoms):
        pos = system.positions[sel_atoms]
        cell = system.cell
        self.make_neighborlist(cutoff=20) # example cutoff
        i, j, S, d = self.nlist.compute(
            points=pos,
            box=cell, 
            periodic=True,
            quantities="ijSd"
        )
        sort_idx = np.lexsort((d, j, i))  # primary i, secondary j, tertiary d

        i_sorted = i[sort_idx]
        j_sorted = j[sort_idx]
        d_sorted = d[sort_idx]
        S_sorted = S[sort_idx]

        mask = np.ones(len(i_sorted), dtype=bool)
        mask[1:] = (i_sorted[1:] != i_sorted[:-1]) | (j_sorted[1:] != j_sorted[:-1])

        i_final = i_sorted[mask]
        j_final = j_sorted[mask]
        d_final = d_sorted[mask]
        S_final = S_sorted[mask]

        return i_final, j_final, S_final, d_final

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
                cum_descriptor = self.descriptor.compute_cumulants(descriptor, self.n_cumulants)
                descriptor_proj = trafo.project(cum_descriptor)
                projected.append(descriptor_proj)
                # TODO:
                #self.ridge.fit(descriptor, descriptor_proj)
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)


    def predict_ridge(self, traj, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float64)
       
        projected_per_type = []

        for idx, trafo in enumerate(self.transformations):
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                cum_descriptor = self.descriptor.compute_cumulants(descriptor, self.n_cumulants)
            
                #ridge_pred = self.ridge[idx].predict(cum_descriptor)
                ridge_pred = self.ridge[idx].predict(descriptor)
                projected.append(ridge_pred)
               
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type


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
        first_soap = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        first_soap_cum = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_cum = np.zeros((first_soap.shape[0], self.interval, first_soap_cum.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_t_cum = np.zeros((len(self.atomsel_element), first_soap_cum.shape[1], first_soap_cum.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_spat = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        sum_soaps_dist = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) #N, S
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    a,b,S,d = self.spatial_correlate(system, atom_type) # return nl indexes of center, neighbor and dist
                    dist = 1/(4.0*np.pi*d**2)
                    sum_soaps_dist[atom_type_idx] += np.sum(dist)
                    sum_soaps_spat[atom_type_idx] += np.einsum('m,mk->k', dist, avg_soap[a,:] + avg_soap[b,:]) / 2  # weighted mean over all pairs
                    cov_t_cum[atom_type_idx] += np.einsum("n, ni,nj->ij", dist , avg_soap[a,:], avg_soap[b,:]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values


        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mu_spat = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov_spat = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            # COV = 1/N ExxT - mumuT
            mu_spat[atom_type_idx] = sum_soaps_spat[atom_type_idx] / sum_soaps_dist[atom_type_idx]
            cov_spat[atom_type_idx] = cov_t_cum[atom_type_idx]/sum_soaps_dist[atom_type_idx] - np.einsum('i,j->ij', mu_spat[atom_type_idx], mu_spat[atom_type_idx])

        self.cov_tot = cov
        return mu, cov_spat, cov


    def fit_ridge_nonincremental(self, traj):
        systems = systems_to_torch(traj, dtype=torch.float64)
        soap_block = self.descriptor.calculate(systems[:1], selected_samples=self.descriptor.selected_samples)
        print(soap_block.shape)
        first_soap = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum() #kernel = delta
        self.ridge = {}


        for idx, trafo in enumerate(self.transformations):
            self.ridge[idx] = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            avg_soap_proj = trafo.project(first_soap) 
            print('avg_soap_proj',avg_soap_proj.shape)
            soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
            avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
            for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                new_soap_values = self.descriptor.calculate([system], selected_samples=self.descriptor.selected_samples)
                cum_soap_values = self.descriptor.compute_cumulants(new_soap_values, self.n_cumulants)
                new_soap_values = None
                if fidx >= self.interval:
                    roll_kernel = np.roll(kernel, fidx%self.interval)
                    # computes a contribution to the correlation function
                    # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                    avg_soap_proj = trafo.project(avg_soap)
                    #print('projshape', avg_soap_proj.shape)
                    #print('nonprog.shape',new_soap_values.shape)
                    soap_values[:,fidx-self.interval,:] = cum_soap_values
                    avg_soaps_projs[:,fidx-self.interval,:] = avg_soap_proj
                buffer[:,fidx%self.interval,:] = cum_soap_values
            print('soapvals',soap_values.dtype)
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            #soap_values=soap_values.reshape((soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2]))
            #avg_soaps_projs=avg_soaps_projs.reshape((avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
            avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
            #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
            #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            self.ridge[idx].fit(soap_values, avg_soaps_projs)
      

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

            torch.save(
                torch.tensor(self.cov1[i]),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_cov1.pt",
            ) 

            torch.save(
                self.descriptor.soap_block.properties.values,
                self.label + f"_center{self.descriptor.centers[i]}" + f"_properties.pt",
            )
