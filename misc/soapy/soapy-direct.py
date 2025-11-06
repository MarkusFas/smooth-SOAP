# Import necessary libraries
import metatensor as mts
import numpy as np
import ase
import ase.io as aseio
from ase.build import molecule
from featomic import SoapPowerSpectrum
import os
from metatensor import Labels, TensorBlock, TensorMap
#from metatrain.gap.model import FPS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import chemiscope
import scipy
from ase import Atoms
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.lib.distances import calc_dihedrals
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  # or load_your_own_data()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

alpha=0.001
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
def pca_model_all(pcaA, pcaE, XA, XE, XA_trf_test, XE_trf_test, XA_clf_test, XE_clf_test, ids_atoms, nstepA, nstepE, pca):
    models = []
    pcA = np.reshape(pcaA, (len(ids_atoms), nstepA, pcaA.shape[-1]))
    pcE = np.reshape(pcaE, (len(ids_atoms), nstepE, pcaE.shape[-1]))
    pcaA_reshape = np.transpose(pcA, (1,0,2))
    pcaE_reshape = np.transpose(pcE, (1,0,2))
    pcaA_flatten = pcaA_reshape.reshape(nstepA, -1)
    pcaE_flatten = pcaE_reshape.reshape(nstepE, -1)
    print(pcA.shape)
    # fit Ridge models for each C


    X_trf_test = np.concatenate((XA_trf_test, XE_trf_test), axis=1)  # Combine test sets for transformation
    yA_trf_test = np.ones(XA_trf_test.shape[1])  # Assuming first half is class A
    yE_trf_test = np.zeros(XE_trf_test.shape[1])  # Assuming second half is class E
    y_trf_test = np.concatenate((yA_trf_test, yE_trf_test))  # Combine labels for test set
    X_clf_test = np.concatenate((XA_clf_test, XE_clf_test), axis=1)  # Combine test sets for classifier
    yA_clf_test = np.ones(XA_clf_test.shape[1])
    yE_clf_test = np.zeros(XE_clf_test.shape[1])
    y_clf_test = np.concatenate((yA_clf_test, yE_clf_test))  # Combine labels for classifier test set
    ridgemodels = []

    X_clf_train = []
    X_final_test = []
    for i, idx in enumerate(ids_atoms):
        #trf = Ridge(alpha=alpha,fit_intercept = False)
        #trf.fit(np.vstack(XA[i]), pcaA_reshape[:,i,:])
        #trf.fit(np.vstack(XE[i]), pcaE_reshape[:,i,:])
        #print('trained E ', i+1)
        #ridgemodels.append(trf)

        #TODO use unseen data XA_trf_test XE_trf_test to get classfier that behaves better on unseen data
        # for now split the test in half!
        X_clf_train.append(pca.transform(X_trf_test[i]))
        X_final_test.append(pca.transform(X_clf_test[i]))

    X_clf_train = np.array(X_clf_train).transpose(1,0,2).reshape(-1,62*3) # (3N,C,P)
    X_final_test = np.array(X_final_test).transpose(1,0,2).reshape(-1,62*3) # (3N,C,P)
    y_clf_train = y_trf_test
    # train model with those train predictions and eval on test predictions
    # Standardize features
    scaler = StandardScaler()
    print('X_clf_train', X_clf_train.shape)
    print('X_clf_test', X_final_test.shape)
    X_clf_train_scaled = scaler.fit_transform(X_clf_train)
    X_final_test_scaled = scaler.transform(X_final_test)

    # Logistic Regression with L1 (Lasso) penalty
    model = LogisticRegression(penalty='l2', solver='liblinear', C=1, random_state=42, fit_intercept=True)
    model.fit(X_clf_train_scaled, y_clf_train)
    y_pred = model.predict(X_final_test_scaled)
    CV_proba = model.predict_proba(X_final_test_scaled)
    
    print('Precision: TP/(TP + FP)')
    print('Recall: TP/P')
    print(classification_report(y_clf_test, y_pred))
    #ll = log_loss(y_true, CV_proba[:,0])
    #print('log loss: ', ll)
    return X_final_test, y_pred, y_clf_test

def get_SOAPPS(traj, ids_atoms, calculator):
    #ids_atoms_surfaces = [30,78,126,174]
    atomselection=Labels(
        names=["atom"],
        values=np.array(np.expand_dims(ids_atoms, axis=1), dtype=np.int32),
    )
    
    #selected_soap = calculator.compute(traj_noli[::], selected_keys=selection, selected_samples=atomselection)
    selected_soap = calculator.compute(traj, selected_keys=selection, selected_samples=atomselection)
    selected_soap = selected_soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

    selected_soap = selected_soap.keys_to_samples(keys_to_move=["center_type"])

    #create tensormap with one block for each atom 
    sliced_selected_soap = {i:mts.slice(selected_soap,axis="samples",
                                 selection=Labels(
        names=["atom",],
        values=np.array([[i]], dtype=np.int32),
    )) for i in ids_atoms}

    timeseries = []
    for i in ids_atoms:
        timeseries.append(sliced_selected_soap[i][0].values[:])

    return np.array(timeseries)


def get_gaussian_avg(timeseries, ids_atoms):
    gaussian_average = {}
    nogaussian_average = {}
    gaussian_average_test = {}
    nogaussian_average_test = {}
    
    for i in ids_atoms:
        gaussian_average[i] =scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
            timeseries[i], sigma=5, axis=0, mode='nearest')#'reflect')
    return gaussian_average

def get_pca(soaps, ids_atoms, pca):
    struct_soap_pca = pca.transform(
    np.vstack([soaps[i] for i in ids_atoms]))
    return struct_soap_pca

def get_pca_direct(soaps, ids_atoms, pca):
    X = np.vstack([soaps[i] for i in ids_atoms]) - pca.mean_
    struct_soap_pca = X @ pca.components_.T  # Project onto PCA components
    return struct_soap_pca

def croptrj(trj):
    trj_out = []
    for atoms in trj:
        trj_out.append(atoms[:196])
    return trj_out


selection = Labels(
    names=["center_type", "neighbor_1_type", "neighbor_2_type"],
    #values=np.array([[8,1,1],[8,8,1],[8,8,8]], dtype=np.int32),
    values=np.array([[i,j,k] for i in [6] for j in [6,7,8] for k in [6,7,8] if j <= k], dtype=np.int32),
)




# load trj only once 
#trajA = aseio.read('prodA.pdb', index='::200')[:]
#print('done with A')
#trajE = aseio.read('prodE.pdb', index='::200')[:]
trajA = aseio.read('prodA-mol.pdb', index=':200')[:]
print('done with A')
trajE = aseio.read('prodE-mol.pdb', index=':200')[:]
print('done with E')
trajA_mol = croptrj(trajA)
trajE_mol = croptrj(trajE)
N = len(trajA)
atoms = trajA[0]
# use only C atoms
ids_atoms = [atom.index for atom in atoms if atom.symbol == 'C']

from sklearn.preprocessing import StandardScaler
HYPER_PARAMETERS = {
    "cutoff": {
        "radius": 5, #4 #5 #6
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 8, #8
        "radial": {"type": "Gto", "max_radial": 8}, #6
    },
}

calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

#soap = calculator.compute(traj)

#timeseriesmetaD = get_SOAPPS(traj_mol, ids_atoms, calculator)
slice1 = int(N*0.6)
slice2 = int(N*0.9)

timeseriesA_train = get_SOAPPS(trajA_mol[:slice1], ids_atoms, calculator)
#np.savetxt('tsA_train.txt', np.array([timeseriesA_train[i] for i in ids_atoms]).transpose(1,0,2).reshape(-1, timeseriesA_train[ids_atoms[0]].shape[-1]))
#timeseriesA_train = None
print('timeseriesA_train')

timeseriesE_train = get_SOAPPS(trajE_mol[:slice1], ids_atoms, calculator)
#np.savetxt('tsE_train.txt', np.array([timeseriesE_train[i] for i in ids_atoms]).transpose(1,0,2).reshape(-1, timeseriesE_train[ids_atoms[0]].shape[-1]))
#timeseriesE_train = None 
print('timeseriesE_train')

timeseriesA_test_trf = get_SOAPPS(trajA_mol[slice1:slice2], ids_atoms, calculator)
#np.savetxt('tsA_test_trf.txt', np.array([timeseriesA_test_trf[i] for i in ids_atoms]).transpose(1,0,2).reshape(-1, timeseriesA_test_trf[ids_atoms[0]].shape[-1]))
#timeseriesA_test_trf = None
print('timeseriesA_testtrf')

timeseriesE_test_trf = get_SOAPPS(trajE_mol[slice1:slice2], ids_atoms, calculator)
#np.savetxt('tsE_test_trf.txt', np.array([timeseriesE_test_trf[i] for i in ids_atoms]).transpose(1,0,2).reshape(-1, timeseriesE_test_trf[ids_atoms[0]].shape[-1]))
#timeseriesE_test_trf = None 
print('timeseriesE_testtrf')

timeseriesA_test_clf = get_SOAPPS(trajA_mol[slice2:], ids_atoms, calculator)
#np.savetxt('tsA_test_clf.txt', np.array([timeseriesA_test_clf[i] for i in ids_atoms]).transpose(1,0,2).reshape(-1, timeseriesA_test_clf[ids_atoms[0]].shape[-1]))
#timeseriesA_test_clf = None
print('timeseriesA_testclf')

timeseriesE_test_clf = get_SOAPPS(trajE_mol[slice2:], ids_atoms, calculator)
#np.savetxt('tsA_test_clf.txt', np.array([timeseriesE_test_clf[i] for i in ids_atoms]).transpose(1,0,2).reshape(-1, timeseriesE_test_clf[ids_atoms[0]].shape[-1]))
#timeseriesE_test_clf = None
print('timeseriesE_testclf')

nstepA = timeseriesA_train.shape[1]
nstepE = timeseriesE_train.shape[1]
y = np.concatenate((np.ones(nstepA), np.zeros(nstepE)))
ids_atoms_i = np.arange(0,len(ids_atoms))

X = np.concatenate((timeseriesA_train, timeseriesE_train), axis=1).transpose(1,0,2)
gaussian_average = get_gaussian_avg(X.transpose(1,0,2), ids_atoms_i)
Xnew = np.vstack([gaussian_average[i] for i in ids_atoms_i])
scaler = StandardScaler()
scaler.fit(Xnew)
X_scaled = scaler.transform(Xnew)
pca=PCA(n_components=3).fit(X_scaled)
print('done with pca fit')
pca_noscale = PCA(n_components=3).fit(Xnew)
print('done with pca noscale fit')

#timeseriesA = X[:nstepA, :, :].transpose(1,0,2)
gaussian_averageA = get_gaussian_avg(timeseriesA_train, ids_atoms_i)

X_scaledA = scaler.transform(np.vstack([gaussian_averageA[i] for i in ids_atoms_i]))
print('scaledA', X_scaledA.shape)
print('nstepA', nstepA)
print(X_scaledA.reshape(62,nstepA,-1).shape)
pcaA_scaled = get_pca(X_scaledA.reshape(62,nstepA,-1), ids_atoms_i, pca)
X_scaledA_raw = scaler.transform(np.vstack([timeseriesA_train[i] for i in ids_atoms_i]))
pcaA_raw_scaled = get_pca(X_scaledA_raw.reshape(62,nstepA,-1), ids_atoms_i, pca)

pcaA = get_pca(gaussian_averageA, ids_atoms_i, pca_noscale)
pcaA_raw = get_pca(timeseriesA_train, ids_atoms_i, pca_noscale)
print('transformed A')
#timeseriesC = get_SOAPPS(trajC_mol, ids_atoms, calculator)
#gaussian_averageC = get_gaussian_avg(timeseriesC)
#pcaC = get_pca(gaussian_averageC, ids_atoms, pca)
#pcaC_raw = get_pca(timeseriesC, ids_atoms, pca)
#print('transformed C')
#timeseriesE = X[nstepA:, :, :].transpose(1,0,2)
gaussian_averageE = get_gaussian_avg(timeseriesE_train, ids_atoms_i)

X_scaledE = scaler.transform(np.vstack([gaussian_averageE[i] for i in ids_atoms_i]))
pcaE_scaled = get_pca(X_scaledE.reshape(62,nstepE,-1), ids_atoms_i, pca)
X_scaledE_raw = scaler.transform(np.vstack([timeseriesE_train[i] for i in ids_atoms_i]))
pcaE_raw_scaled = get_pca(X_scaledE_raw.reshape(62,nstepE,-1), ids_atoms_i, pca)

pcaE = get_pca(gaussian_averageE, ids_atoms_i, pca_noscale)
pcaE_raw = get_pca(timeseriesE_train, ids_atoms_i, pca_noscale)
print('transformed E')


print('avg2avg')
Xridge_avg2avg, yridge_avg2avg, yridge_avg2avg_true = ridge_model_all(pcaA, pcaE, gaussian_averageA, gaussian_averageE, timeseriesA_test_trf, timeseriesE_test_trf, timeseriesA_test_clf, timeseriesE_test_clf, ids_atoms_i, nstepA, nstepE, pca)
print('avg2raw')
Xridge_avg2raw, yridge_avg2raw, yridge_avg2raw_true = ridge_model_all(pcaA_raw, pcaE_raw, gaussian_averageA, gaussian_averageE, timeseriesA_test_trf, timeseriesE_test_trf, timeseriesA_test_clf, timeseriesE_test_clf, ids_atoms_i, nstepA, nstepE, pca)
print('raw2raw')
Xridge_raw2raw, yridge_raw2raw, yridge_raw2raw_true = ridge_model_all(pcaA_raw,
                                                                      pcaE_raw,
                                                                      timeseriesA_train,
                                                                      timeseriesE_train,
                                                                      timeseriesA_test_trf, 
                                                                      timeseriesE_test_trf, 
                                                                      timeseriesA_test_clf, 
                                                                      timeseriesE_test_clf, 
                                                                      ids_atoms_i, 
                                                                      nstepA, 
                                                                      nstepE,
                                                                      pca
                                                                      )
print('raw2avg')
Xridge_raw2avg, yridge_raw2avg, yridge_raw2avg_true = ridge_model_all(pcaA,
                                                                      pcaE,
                                                                      timeseriesA_train, 
                                                                      timeseriesE_train, 
                                                                      timeseriesA_test_trf, 
                                                                      timeseriesE_test_trf, 
                                                                      timeseriesA_test_clf, 
                                                                      timeseriesE_test_clf, 
                                                                      ids_atoms_i, 
                                                                      nstepA, 
                                                                      nstepE,
                                                                      pca
                                                                      )

cwdA = '/Users/markusfasching/EPFL/Work/project-CVs/cyclosporin/gromacs/cycloA/'
uA = mda.Universe(os.path.join(cwdA, "prod.tpr"), os.path.join(cwdA, "prod.xtc"))
Cs = uA.select_atoms('element C')
feature_names = [f'{C.name} from {C.resname} PCA {i+1}' for C in Cs for i in range(3)]
feature_names_C = [f'{C.name} from {C.resname}' for C in Cs]
pca_raw = np.concatenate((pcaA_raw.reshape(-1,62,3), pcaE_raw.reshape(-1,62,3))).reshape(-1,3*62)
pca_avg = np.concatenate((pcaA.reshape(-1,62,3), pcaE.reshape(-1,62,3))).reshape(-1,3*62)
fig, ax = plt.subplots(62,6, figsize=(18, 200))
mins = np.min(np.concatenate((Xridge_avg2avg, Xridge_avg2raw, Xridge_raw2avg, Xridge_raw2raw)), axis=0)
maxs = np.max(np.concatenate((Xridge_avg2avg, Xridge_avg2raw, Xridge_raw2avg, Xridge_raw2raw)), axis=0)
for i in range(len(ids_atoms)):
    ax[i,0].set_title(f'{i+1} {feature_names_C[i]} time avg')
    ax[i,1].set_title(f'{i+1} {feature_names_C[i]} ridge time avg')
    ax[i,2].set_title(f'{i+1} avg2avg')
    ax[i,3].set_title(f'{i+1} avg2raw')
    ax[i,4].set_title(f'{i+1} raw2raw')
    ax[i,5].set_title(f'{i+1} raw2avg')
    #if i in coef_:
    #    ax[i,0].set_title(f'{feature_names_C[i]} PCA from avg', color='red')
    ##else:
    #    ax[i,0].set_title(f'{feature_names_C[i]} PCA from avg')
    pca1_i = i*3
    pca2_i = i*3 + 1
    ax[i,0].scatter(pca_avg[:,pca1_i], pca_avg[:,pca2_i], c=y, cmap='coolwarm', alpha=0.5, label=' time avg') #C=50
    ax[i,1].scatter(pca_raw[:,pca1_i], pca_raw[:,pca2_i], c=y, cmap='coolwarm', alpha=0.5, label='ridge time avg') #C=50
    ax[i,2].scatter(Xridge_avg2avg[:,pca1_i], Xridge_avg2avg[:,pca2_i], c=yridge_avg2avg, cmap='coolwarm', alpha=0.5, label=' raw') #C=50
    ax[i,3].scatter(Xridge_avg2raw[:,pca1_i], Xridge_avg2raw[:,pca2_i], c=yridge_avg2raw, cmap='coolwarm', alpha=0.5, label='ridge raw') #C=50
    ax[i,4].scatter(Xridge_raw2raw[:,pca1_i], Xridge_raw2raw[:,pca2_i], c=yridge_raw2raw, cmap='coolwarm', alpha=0.5, label='ridge raw2raw') #C=50
    ax[i,5].scatter(Xridge_raw2avg[:,pca1_i], Xridge_raw2avg[:,pca2_i], c=yridge_raw2avg_true, cmap='coolwarm', alpha=0.5, label='ridge raw2avg') #C=50
    for j in range(6):
        ax[i,j].set_xlim(mins[pca1_i], maxs[pca1_i])
        ax[i,j].set_ylim(mins[pca2_i], maxs[pca2_i])
       
    
    #ax[i,2].scatter(Xridge_raw_scaled[:,pca1_i], Xridge_raw_scaled[:,pca2_i], c=y_raw_scaled, cmap='coolwarm', alpha=0.5, label='scaled raw') #C=50
    #ax[i,3].scatter(Xridge_raw[:,pca1_i], Xridge_raw[:,pca2_i], c=y_raw, cmap='coolwarm', alpha=0.5, label='raw') #C=50

plt.savefig('pcaplots.png')

