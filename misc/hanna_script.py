# Import necessary libraries
import metatensor.torch as mts
import numpy as np
import ase
import ase.io as aseio
from ase.build import molecule
from featomic.torch import SoapPowerSpectrum

import os

from metatensor.torch import Labels, TensorBlock, TensorMap
#from metatrain.gap.model import FPS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import chemiscope
import scipy
from ase import Atoms

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import torch
import itertools #import combinations
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    systems_to_torch,
)

def slice_features(selected_soap,ids_atoms):
    sliced_selected_soap = {}
    for iatom in ids_atoms:
        sliced_selected_soap[iatom] = mts.slice(selected_soap,axis="samples",
                                     selection=Labels(
        names=["atom",],
        values=torch.tensor([[iatom]]),
    ))  
    return sliced_selected_soap

def time_averaged_features(sliced_features, ids_atoms, sigma=5):
    gaussian_average = {}
    for i in ids_atoms:
        gaussian_average[str(i)] =scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
            sliced_features[i][0].values.numpy(), sigma=sigma, axis=0, mode='nearest')#'reflect')
    return gaussian_average

def just_features(sliced_features, ids_atoms, sigma=5):
    nogaussian_average = {}
    for i in ids_atoms:
        nogaussian_average[str(i)] =sliced_features[i][0].values[:]
    return nogaussian_average



traj = aseio.read('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/interfaces/250_275_fast/positions.lammpstrj',':200')#../data/water_ice_md_230K/positions.dump','-200::1')[:] #100
test_traj = aseio.read('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/interfaces/250_275_fast/positions.lammpstrj',':200')[:] #100
print(len(traj))
systems = systems_to_torch(traj, dtype=torch.float64)
test_systems = systems_to_torch(test_traj, dtype=torch.float64)

stride=2

ids_atoms_O = [i.index for i in traj[0] if i.number==8][:]
ids_atoms_ice=[i.index for i in traj[0] if (i.number==8 and 34>i.position[2]>22 and i.index in ids_atoms_O)]
ids_atoms_water=[i.index for i in traj[0] if (i.number==8 and 14>i.position[2]>4 and i.index in ids_atoms_O)]
ids_atoms_inter=[i.index for i in traj[0] if (i.number==8 and i.index not in ids_atoms_ice+ids_atoms_water and i.index in ids_atoms_O)]
ids_atoms = ids_atoms_O

#envidx=np.array(['water' if (frames[i][j].number==atomtype and j in ids_atoms_water) else 'ice' if (frames[i][j].number==atomtype and j in ids_atoms_ice ) else 'inter' if frames[i][j].number==atomtype  for i in range(len(frames)) for j in range(len(frames[i])) ])
#print(envidx)

test_ids_atoms_O = [i.index for i in test_traj[0] if i.number==8][:]
test_ids_atoms_ice=[i.index for i in test_traj[0] if (i.number==8 and 34>i.position[2]>22 and i.index in test_ids_atoms_O)]
test_ids_atoms_water=[i.index for i in test_traj[0] if (i.number==8 and 14>i.position[2]>4 and i.index in test_ids_atoms_O)]
test_ids_atoms_inter=[i.index for i in test_traj[0] if (i.number==8 and i.index not in test_ids_atoms_ice+test_ids_atoms_water and i.index in test_ids_atoms_O)]
test_ids_atoms = ids_atoms_O

print('trainsetlens',len(ids_atoms_O), len(ids_atoms_ice), len(ids_atoms_water), len(ids_atoms_inter))
print('testsetlens',len(test_ids_atoms_O), len(test_ids_atoms_ice), len(test_ids_atoms_water), len(test_ids_atoms_inter))

selection = Labels(
    names=["center_type", "neighbor_1_type", "neighbor_2_type"],
    values=torch.tensor([[8, 1, 1], [8, 1, 8], [8, 8, 8]]),
)

cutoffs=[7]#,8,10]#[3,4,5,6,7,8,10]
angular_max=[8]#[1,2,3,4,5,6,8]
radial_max=[6]#[1,2,3,4,5,6,8]
combis=list(itertools.product(cutoffs, angular_max, radial_max))
print('combis',combis)

for combi in combis:
    print('now starting combi', combi)


    cutoff=combi[0]
    amax=combi[1]
    rmax=combi[2]

#    if os.path.isfile('PCA_{}_{}_{}.png'.format(cutoff,amax,rmax)):
#        'exists already'
#        continue

    HYPER_PARAMETERS = {
        "cutoff": {
            "radius": cutoff, #4 #5 #6
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3,
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": amax, #8
            "radial": {"type": "Gto", "max_radial": rmax}, #6
        },
    }
    
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    
    atomselection=Labels(
        names=["atom"],
        values=torch.tensor(np.expand_dims(ids_atoms, axis=1)),
    )
    
    selected_soap = calculator.compute(systems, selected_keys=selection, selected_samples=atomselection)
    selected_soap = selected_soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])    
    selected_soap = selected_soap.keys_to_samples(keys_to_move=["center_type"])
     
    sliced_features= slice_features(selected_soap,ids_atoms)
    gaussian_average= time_averaged_features(sliced_features, ids_atoms, sigma=5)
    nogaussian_average= just_features(sliced_features, ids_atoms, sigma=5)

    test_atomselection=Labels(
        names=["atom"],
        values=torch.tensor(np.expand_dims(test_ids_atoms, axis=1)),
    )
    test_selected_soap = calculator.compute(test_systems[::], selected_keys=selection, selected_samples=test_atomselection)
    test_selected_soap = test_selected_soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])    
    test_selected_soap = test_selected_soap.keys_to_samples(keys_to_move=["center_type"])
    test_sliced_features= slice_features(test_selected_soap,test_ids_atoms)
    test_gaussian_average= time_averaged_features(test_sliced_features, test_ids_atoms, sigma=5)
    test_nogaussian_average= just_features(test_sliced_features, test_ids_atoms, sigma=5)

#
#    sliced_selected_soap = {}
#    for i in ids_atoms:
#        sliced_selected_soap[i] = mts.slice(selected_soap,axis="samples",
#                                     selection=Labels(
#        names=["atom",],
#        values=np.array([[i]], dtype=np.int32),
#    ))
#    
#    gaussian_average = {}
#    nogaussian_average = {}
#    for i in ids_atoms:
#        gaussian_average[i] =scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
#            sliced_selected_soap[i][0].values[:], sigma=5, axis=0, mode='nearest')#'reflect')
#        nogaussian_average[i] =sliced_selected_soap[i][0].values[:]
    
    pca=PCA(n_components=3).fit(np.vstack([gaussian_average[str(i)] for i in ids_atoms]))
    struct_soap_pca = pca.transform(
        np.vstack([gaussian_average[str(i)] for i in ids_atoms]))
    struct_soap_pca_nonav = pca.transform(
        np.vstack([nogaussian_average[str(i)] for i in ids_atoms]))
    test_struct_soap_pca = pca.transform(
        np.vstack([test_gaussian_average[str(i)] for i in test_ids_atoms]))
    test_struct_soap_pca_nonav = pca.transform(
        np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms]))
     
    values_av=np.vstack([gaussian_average[str(i)] for i in ids_atoms])
    values_nonav=np.vstack([nogaussian_average[str(i)] for i in ids_atoms])
    test_values_av=np.vstack([test_gaussian_average[str(i)] for i in test_ids_atoms])
    test_values_nonav=np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms])
    
    #RIDGE avtoav 
    alpha=0.3
    clf = Ridge(alpha=alpha,fit_intercept = False)
    clf.fit(np.vstack(values_av), struct_soap_pca[:,:2])
    #RIDGE nontonon 
    clf_nontonon = Ridge(alpha=alpha,fit_intercept = False)
    clf_nontonon.fit(np.vstack(values_nonav), struct_soap_pca_nonav[:,:2])
    #RIDGE nontoav 
    clf_nontoav = Ridge(alpha=alpha,fit_intercept = False)
    clf_nontoav.fit(np.vstack(values_nonav), struct_soap_pca[:,:2])
    #RIDGE avtonon
    clf_avtonon = Ridge(alpha=alpha,fit_intercept = False)
    clf_avtonon.fit(np.vstack(values_av), struct_soap_pca_nonav[:,:2])


    def plot_data(clf, values_av,values_nonav, nogaussian_average,test_nogaussian_average, struct_soap_pca, name='av', ids_atoms=ids_atoms,ids_atoms_inter=ids_atoms_inter,ids_atoms_water=ids_atoms_water,ids_atoms_ice=ids_atoms_ice):    
        pca_nonav=np.dot(values_nonav, clf.coef_.T)
        pca_av=np.dot(values_av, clf.coef_.T)
    
        ti=np.dot(np.vstack([nogaussian_average[str(i)] for i in ids_atoms]),clf.coef_.T)
        ti_inter=np.dot(np.vstack([nogaussian_average[str(i)] for i in ids_atoms_inter]),clf.coef_.T)
        ti_water=np.dot(np.vstack([nogaussian_average[str(i)] for i in ids_atoms_water]),clf.coef_.T)
        ti_ice=np.dot(np.vstack([nogaussian_average[str(i)] for i in ids_atoms_ice]),clf.coef_.T)
#        ri=clf.predict(np.vstack([nogaussian_average[i] for i in ids_atoms]))
        test_values_nonav=np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms])
        test_pca_nonav=np.dot(test_values_nonav, clf.coef_.T)
        
#        envidx=np.array(['w' for i in range(len(frames)) for j in range(len(frames[i])) if (frames[i][j].number==atomtype and j in ids_atoms_water) else 'ice' if (frames[i][j].number==atomtype and j in ids_atoms_ice ) else 'inter' if frames[i][j].number==atomtype ])
#        print(envidx)

        s=1
    
        #av PCA      Ridge1
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
#        cmap = LinearSegmentedColormap.from_list('mycmap', colors=['r','darkorange','blue'], N=3)
#        classifier=GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
#        classifier = make_pipeline(StandardScaler(), classifier)
#        classifier.fit(pca_nonav,tilattice)
#        score = classifier.score(test_pca_nonav,test_tilattice)
#        DecisionBoundaryDisplay.from_estimator(
#            classifier, test_pca_nonav, alpha=0.3,cmap=cmap, ax=ax[1], eps=0.01
#        )
#        #print('score', score)
#        ax[1].text(0.1,0.15,'Score: {}'.format(score),  transform=ax[1].transAxes)
        alpha=0.1
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        #ax.plot(struct_soap_pca[:,0],struct_soap_pca[:,1],'o',markersize=1,color='grey',label='Averaged SOAP PCA')
        ax.plot(ti_inter[:,0],ti_inter[:,1],'o',markersize=1,alpha=alpha, color='green', marker='x', label='Interface')#Ridge fit of not averaged SOAP to PC1/PCA2')
        ax.plot(ti_water[:,0],ti_water[:,1],'o',markersize=1,alpha=alpha, color='r', label='water')#'Ridge fit of not averaged SOAP to PC1/PCA2')
        ax.plot(ti_ice[:,0],ti_ice[:,1],'o',markersize=1,alpha=alpha, color='blue', label='ice')#'Ridge fit of not averaged SOAP to PC1/PCA2')
    
        ax.legend(loc='upper right')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.text(0.1,0.1,'Feature length: {}'.format(len(clf.coef_.T)),  transform=ax.transAxes)
        plt.savefig('PCA_{}_{}_{}_{}.png'.format(cutoff,amax,rmax,name),dpi=300)
    
        torch.save(torch.tensor(clf.coef_),'water_ridge_matrix_{}{}{}_{}.pt'.format(cutoff,amax,rmax,name))
    #    print('saved')
        plt.close('all')

    def plot_data_direct(pca, values_av,values_nonav, nogaussian_average,test_nogaussian_average, struct_soap_pca, name='av', ids_atoms=ids_atoms,ids_atoms_inter=ids_atoms_inter,ids_atoms_water=ids_atoms_water,ids_atoms_ice=ids_atoms_ice):    

        ti = pca.transform(
            np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms]))
        ti_inter = pca.transform(
            np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms_inter]))
        ti_water = pca.transform(
            np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms_water]))
        ti_ice = pca.transform(
            np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms_ice]))


#        ri=clf.predict(np.vstack([nogaussian_average[i] for i in ids_atoms]))
        test_values_nonav=np.vstack([test_nogaussian_average[str(i)] for i in test_ids_atoms])
        test_pca_nonav=np.dot(test_values_nonav, clf.coef_.T)
        
#        envidx=np.array(['w' for i in range(len(frames)) for j in range(len(frames[i])) if (frames[i][j].number==atomtype and j in ids_atoms_water) else 'ice' if (frames[i][j].number==atomtype and j in ids_atoms_ice ) else 'inter' if frames[i][j].number==atomtype ])
#        print(envidx)

        s=1
    
        #av PCA      Ridge1
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
#        cmap = LinearSegmentedColormap.from_list('mycmap', colors=['r','darkorange','blue'], N=3)
#        classifier=GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
#        classifier = make_pipeline(StandardScaler(), classifier)
#        classifier.fit(pca_nonav,tilattice)
#        score = classifier.score(test_pca_nonav,test_tilattice)
#        DecisionBoundaryDisplay.from_estimator(
#            classifier, test_pca_nonav, alpha=0.3,cmap=cmap, ax=ax[1], eps=0.01
#        )
#        #print('score', score)
#        ax[1].text(0.1,0.15,'Score: {}'.format(score),  transform=ax[1].transAxes)
        alpha=0.1
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        #ax.plot(struct_soap_pca[:,0],struct_soap_pca[:,1],'o',markersize=1,color='grey',label='Averaged SOAP PCA')
        ax.plot(ti_inter[:,0],ti_inter[:,1],'o',markersize=1,alpha=alpha, color='green', marker='x', label='Interface')#Ridge fit of not averaged SOAP to PC1/PCA2')
        ax.plot(ti_water[:,0],ti_water[:,1],'o',markersize=1,alpha=alpha, color='r', label='water')#'Ridge fit of not averaged SOAP to PC1/PCA2')
        ax.plot(ti_ice[:,0],ti_ice[:,1],'o',markersize=1,alpha=alpha, color='blue', label='ice')#'Ridge fit of not averaged SOAP to PC1/PCA2')
    
        ax.legend(loc='upper right')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.text(0.1,0.1,'Feature length: {}'.format(len(clf.coef_.T)),  transform=ax.transAxes)
        plt.savefig('direct_PCA_{}_{}_{}_{}.png'.format(cutoff,amax,rmax,name),dpi=300)
    
        torch.save(torch.tensor(clf.coef_),'water_ridge_matrix_{}{}{}_{}.pt'.format(cutoff,amax,rmax,name))
    #    print('saved')
        plt.close('all')

    #plot_data(clf, values_av,values_nonav, nogaussian_average,test_nogaussian_average, struct_soap_pca, name='av')
    #plot_data(clf_nontonon, values_av,values_nonav, nogaussian_average,test_nogaussian_average, struct_soap_pca, name='nontonon')
    #plot_data(clf_nontoav, values_av,values_nonav, nogaussian_average,test_nogaussian_average, struct_soap_pca, name='nontoav')
    #plot_data(clf_avtonon, values_av,values_nonav, nogaussian_average,test_nogaussian_average, struct_soap_pca, name='avtonon')
    plot_data_direct(pca, values_av,values_nonav, nogaussian_average,test_nogaussian_average, struct_soap_pca, name='avtonon')
    
#    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#    #scatter = ax.scatter(struct_soap_pca[:, 0], struct_soap_pca[:, 1], c=np.array([colordict[i] for i in tilattice]).ravel().tolist(), alpha=0.2,edgecolor='none')
#    #scatter_av = ax.scatter(pca_av[:, 0], pca_av[:, 1], c=np.array([colordict[i] for i in tilattice]).ravel().tolist(), alpha=0.2, marker='x')
#    CV1_pcaav=[]
#    CV2_pcaav=[]
#    CV1_av=[]
#    CV2_av=[]
#    CV1_nonav=[]
#    CV2_nonav=[]
#    nro=len([i for i in traj[0] if i.number==8])
##    print('nr ti', nro)
#    for n in range(int(len(struct_soap_pca)/nro)):
##          print('areas: -n- ',n, nro*n,nro*n+nro)
#          CV1_pcaav.append(np.sum(struct_soap_pca[nro*n:nro*n+nro,0]))  
#          CV2_pcaav.append(np.sum(struct_soap_pca[nro*n:nro*n+nro,1]))  
#          CV1_av.append(np.sum(pca_av[nro*n:nro*n+nro,0]))  
#          CV2_av.append(np.sum(pca_av[nro*n:nro*n+nro,1]))  
#          CV1_nonav.append(np.sum(pca_nonav[nro*n:nro*n+nro,0]))  
#          CV2_nonav.append(np.sum(pca_nonav[nro*n:nro*n+nro,1]))  
#    scatter = ax.scatter(CV1_pcaav, CV2_pcaav,alpha=0.2,edgecolor='none', label='CVpca')#,  c=np.array([colordict[i] for i in tilattice[::nro]]).ravel().tolist(), alpha=0.2,edgecolor='none')
#    scatter = ax.scatter(CV1_av, CV2_av,  alpha=0.2,marker='d', label='CVav')
#    scatter = ax.scatter(CV1_nonav, CV2_nonav, alpha=0.2,marker='x', label='CVnonav')
#
#    ax.set_xlabel("PCA[1]")
#    ax.set_ylabel("PCA[2]")
#    ax.legend()
#
#    #plt.show()
#
#    ax.text(0.1,0.1,'Feature length: {}'.format(len(clf.coef_.T)),  transform=ax.transAxes)
#    #ax.legend()#handles=legend_elements)
#    plt.savefig('PCA_CVsum_{}_{}_{}.png'.format(cutoff,amax,rmax),dpi=300)
#    #plt.savefig('PCA_{}_{}_{}_23.png'.format(cutoff,amax,rmax),dpi=100)
#
#    plt.close()
#









