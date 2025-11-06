# Import necessary libraries
import metatensor as mts
import numpy as np
import ase
import ase.io as aseio
from ase.build import molecule
from featomic.torch import SoapPowerSpectrum
import os
from metatensor import Labels, TensorBlock, TensorMap
from metatensor.torch import Labels as Labels_torch
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
from sklearn.metrics import brier_score_loss
from skmatter.decomposition import PCovR
alpha=0.01
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from scipy.linalg import eigh
import torch
from metatomic.torch import systems_to_torch

def lda_model_all(XA, XB, X_test, NCOMPONENTS=4):
    
    # LDA
    N, T, S = XA.shape
    eps = 1e-10  # numerical stability

    # Per-atom means: shape (N, S)
    mu_A = XA.mean(axis=1)
    mu_B = XB.mean(axis=1)

    # Per-atom between-class scatter: shape (N, S, S)
    delta = mu_A - mu_B  # (N, S)
    S_B_all = delta[:, :, None] @ delta[:, None, :]  # outer product per atom
    S_B = S_B_all.mean(axis=0)  # average over atoms

    # Per-atom centered data
    X_A_centered = XA - mu_A[:, None, :]  # (N, T, S)
    X_B_centered = XB - mu_B[:, None, :]  # (N, T, S)

    # Compute per-atom covariances using einsum
    cov_A = np.einsum('nts,ntk->nks', X_A_centered, X_A_centered) / (T - 1)
    cov_B = np.einsum('nts,ntk->nks', X_B_centered, X_B_centered) / (T - 1)

    # Per-atom within-class scatter: sum of covariances
    S_W_all = cov_A + cov_B  # shape: (N, S, S)
    S_W = S_W_all.mean(axis=0)

    # Solve generalized eigenvalue problem
    eigvals = [None for _ in range(S_W_all.shape[0])]
    eigvecs = [None for _ in range(S_W_all.shape[0])]
    for i, S_W in enumerate(S_W_all):
        eigvals[i], eigvecs[i] = eigh(S_B_all[i], S_W + eps * np.eye(S))  # add small regularizer

    # Sort in descending order
    eigvals = np.array(eigvals)[:, ::-1]
    eigvecs = np.array(eigvecs)[:, :, ::-1]

    # Project test data
    mu_global = 0.5 * (mu_A.mean(axis=0) + mu_B.mean(axis=0))  # (S,)
    #X_test_centered = X_test - mu_global[None, None, :]        # (N, T, S)
    projected_test = np.einsum('nts,nsk->ntk', X_test, eigvecs[:, :, :NCOMPONENTS])  # (N, T, k)
    torch.save(torch.from_numpy(eigvecs.copy()),f'old_lda_eigvecs_allC_{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')
    return eigvals, eigvecs, projected_test, S_B, S_W


def pca_model_all(XA, XB, X_test, NCOMPONENTS=4):
    
    N, T, S = XA.shape
    eps = 1e-10

    # 1. Concatenate along time axis → (N, 2T, S)
    X_train = np.concatenate([XA, XB], axis=1)

    # 2. Per-atom mean: shape (N, S)
    mu = X_train.mean(axis=1)

    # 3. Centered data
    X_centered = X_train - mu[:, None, :]  # (N, 2T, S)

    # 4. Per-atom covariance matrices using einsum
    covariances = np.einsum('nts,ntk->nks', X_centered, X_centered) / (2*T - 1)

    # 5. Average over atoms
    #avg_cov = covariances.mean(axis=0)

    # 6. Eigenvalue decomposition of covariance matrix
    eigvals = [None for _ in range(covariances.shape[0])]
    eigvecs = [None for _ in range(covariances.shape[0])]
    for n, avg_cov in enumerate(covariances):
        eigvals[n], eigvecs[n] = eigh(avg_cov + eps * np.eye(S))  # Regularize

    # Sort eigenvectors by descending eigenvalue
    eigvals = np.array(eigvals)[:, ::-1]
    eigvecs = np.array(eigvecs)[:, :, ::-1]

    # 7. Project centered test data: center using mean of train data
    mu_global = mu.mean(axis=0)  # (S,)
    #X_test_centered = X_test - mu_global[None, None, :]
    projected_test = np.einsum('nts,nsk->ntk', X_test, eigvecs[:, :, :NCOMPONENTS])

    return eigvals, eigvecs, projected_test, avg_cov

def lda_model_all_meanC_ridge(XA, XB, X_test,AVG_IND=True, SIGMA=50, NCOMPONENTS=4):
    print('XAbefore', XA.shape)
    XA_pre = XA.copy()
    XB_pre = XB.copy()
    # LDA
    NA, TA, SA = XA.shape
    NB, TB, SB = XB.shape
    Ntest, Ttest, Stest = X_test.shape
    eps = 1e-10  # numerical stability
    # Per-atom means: shape (N*T, S)
    if AVG_IND:
        print('timeavg done')
        XA_tavg = scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
                XA, sigma=SIGMA, axis=1, mode='nearest')#'reflect')
        XB_tavg = scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
                XB, sigma=SIGMA, axis=1, mode='nearest')#'reflect')
    else:
        XA_tavg = XA
        XB_tavg = XB
    
    XA = XA.reshape(-1, XA.shape[-1])
    XB = XB.reshape(-1, XA.shape[-1])
    XA_tavg = XA_tavg.reshape(-1, XA.shape[-1])
    XB_tavg = XB_tavg.reshape(-1, XA.shape[-1])
    mu_A = XA_tavg.mean(axis=0)
    mu_B = XB_tavg.mean(axis=0)

    # Per-atom between-class scatter: shape (S, S)
    delta = mu_A - mu_B  # (S)
    S_B_all = delta[:, None] @ delta[None, :]  # outer product per atom
    #S_B = S_B_all.mean(axis=0)  # average over atoms

    # Per-atom centered data
    X_A_centered = XA_tavg - mu_A  # (N*T, S)
    X_B_centered = XB_tavg - mu_B  # (N*T, S)

    # Compute per-atom covariances using einsum
    cov_A = np.einsum('ts,tk->ks', X_A_centered, X_A_centered) / (TA - 1) #S,S
    cov_B = np.einsum('ts,tk->ks', X_B_centered, X_B_centered) / (TB - 1)

    # within-class scatter: sum of covariances
    S_W_all = cov_A + cov_B  # shape: (N, S, S)
    #S_W = S_W_all.mean(axis=0)

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(S_B_all, S_W_all + eps * np.eye(SA))  # add small regularizer

    # Sort in descending order
    
    eigvals = np.array(eigvals)[::-1]
    eigvecs = np.array(eigvecs)[:, ::-1]
    
    trf = Ridge(alpha=alpha,fit_intercept = False)
    ldaA = np.einsum('ts,sk->tk', XA, eigvecs[:, :NCOMPONENTS])
    ldaB = np.einsum('ts,sk->tk', XB, eigvecs[:, :NCOMPONENTS])
    trf.fit(XA, ldaA)
    trf.fit(XB, ldaB)

    # Project test data
    mu_global = (mu_A*NA*TA + mu_B*NB*TB)/(NA*TA + NB*TB)  # (S,)
    X_test_centered = X_test.reshape(-1, X_test.shape[-1]) - mu_global       # (N, T, S)
    projected_test = np.einsum('ts,sk->tk', X_test_centered, eigvecs[:, :NCOMPONENTS])  # (N, T, k)
    if AVG_IND:
        torch.save(torch.from_numpy(eigvecs.copy()).to(torch.float64),f'lda_eigvecs_meanC{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')
        torch.save(torch.from_numpy(mu_global.copy()).to(torch.float64),f'lda_mu_meanC{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')
        torch.save(torch.from_numpy(eigvals.copy()).to(torch.float64),f'lda_eigval_meanC{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')
    else:
        torch.save(torch.from_numpy(eigvecs.copy()).to(torch.float64),f'lda_eigvecs_meanC{cutoff}{maxang}{maxrad}_sig{0}.pt')
        torch.save(torch.from_numpy(mu_global.copy()).to(torch.float64),f'lda_mu_meanC{cutoff}{maxang}{maxrad}_sig{0}.pt')
        torch.save(torch.from_numpy(eigvals.copy()).to(torch.float64),f'lda_eigval_meanC{cutoff}{maxang}{maxrad}_sig{0}.pt')


    fig, axs = plt.subplots(2,1, figsize=(8,12))
    X = np.concatenate((XA_pre, XB_pre), axis=1)
    ldaA = np.einsum('ts,sk->tk', XA_pre[0] - mu_A, eigvecs[:, :NCOMPONENTS])
    ldaB = np.einsum('ts,sk->tk', XB_pre[0] - mu_B, eigvecs[:, :NCOMPONENTS])
    LDAX = np.concatenate((ldaA, ldaB), axis=0).T
    print('XT', X.T.shape)
    print('XA', X_A_centered.shape)
    print('LDAX', LDAX.shape)
    for i, col in enumerate(X.transpose(1,0,2).reshape(X.shape[1], -1)):
        axs[0].plot(col, alpha=0.1)
    for col in LDAX:
        axs[1].plot(col, alpha=0.1)
    plt.savefig('lda-plot-timeseries.png')
    X_pred = trf.predict(X_test.reshape(-1, X_test.shape[-1]))
    
    #torch.save(torch.from_numpy(trf
    return eigvals, eigvecs, projected_test.reshape(Ntest, Ttest, -1), S_B_all, S_W_all
    #return eigvals, eigvecs, X_pred.reshape(Ntest, Ttest, -1), S_B_all, S_W_all

def lda_model_all_meanC(XA, XB, X_test, NCOMPONENTS=4):
    
    # LDA
    N, T, S = XA.shape
    eps = 1e-10  # numerical stability
    XA = XA.reshape(-1, XA.shape[-1])
    XB = XB.reshape(-1, XA.shape[-1])
    # Per-atom means: shape (N*T, S)
    mu_A = XA.mean(axis=0)
    mu_B = XB.mean(axis=0)

    # Per-atom between-class scatter: shape (S, S)
    delta = mu_A - mu_B  # (S)
    S_B_all = delta[:, None] @ delta[None, :]  # outer product per atom
    #S_B = S_B_all.mean(axis=0)  # average over atoms

    # Per-atom centered data
    X_A_centered = XA - mu_A  # (N*T, S)
    X_B_centered = XB - mu_B  # (N*T, S)

    # Compute per-atom covariances using einsum
    cov_A = np.einsum('ts,tk->ks', X_A_centered, X_A_centered) / (T - 1) #S,S
    cov_B = np.einsum('ts,tk->ks', X_B_centered, X_B_centered) / (T - 1)

    # within-class scatter: sum of covariances
    S_W_all = cov_A + cov_B  # shape: (N, S, S)
    #S_W = S_W_all.mean(axis=0)

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(S_B_all, S_W_all + eps * np.eye(S))  # add small regularizer

    # Sort in descending order
    
    eigvals = np.array(eigvals)[::-1]
    eigvecs = np.array(eigvecs)[:, ::-1]

    # Project test data
    mu_global = 0.5 * (mu_A.mean(axis=0) + mu_B.mean(axis=0))  # (S,)
    #X_test_centered = X_test - mu_global[None, None, :]        # (N, T, S)
    projected_test = np.einsum('nts,sk->ntk', X_test, eigvecs[:, :NCOMPONENTS])  # (N, T, k)
    torch.save(torch.from_numpy(eigvecs.copy()), 'lda_eigvecs_meanC.pt')
    return eigvals, eigvecs, projected_test, S_B_all, S_W_all

def pca_model_all_meanC_ridge(XA, XB, X_test,AVG_IND=True, SIGMA=50, NCOMPONENTS=4):
    
    # PCA
    N, T, S = XA.shape
    Ntest, Ttest, Stest = X_test.shape
    eps = 1e-10  # numerical stability
    # Per-atom means: shape (N*T, S)
    X_train = np.concatenate([XA, XB], axis=1)
    if AVG_IND:
        X_tavg = scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
                X_train, sigma=SIGMA, axis=1, mode='nearest')#'reflect')
    else:
        X_tavg = X_train
        
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_tavg = X_tavg.reshape(-1, X_tavg.shape[-1])
    # 2. Per-atom mean: shape (S)
    mu = X_tavg.mean(axis=0)

    # 3. Centered data
    X_centered = X_tavg - mu# (2T*N, S)

    # 4. Per-atom covariance matrices using einsum
    covariances = np.einsum('ts,tk->ks', X_centered, X_centered) / (2*T*N - 1)

    # 5. Average over atoms
    #avg_cov = covariances.mean(axis=0)

    # 6. Eigenvalue decomposition of covariance matrix
    eigvals, eigvecs = eigh(covariances + eps * np.eye(S))  # Regularize

    # Sort eigenvectors by descending eigenvalue
    eigvals = np.array(eigvals)[::-1]
    eigvecs = np.array(eigvecs)[:, ::-1]
    
    
    pca = PCA(n_components=4)
    X_pca = pca.fit(X_tavg)
    X_pca = pca.transform(X_train)
    
    #trf = Ridge(alpha=alpha,fit_intercept = False)
    #pcatrain = np.einsum('ts,sk->tk', X_train, eigvecs[:, :NCOMPONENTS])
    #trf.fit(X_train, pcatrain)

    #X_test_centered = X_test - mu_global[None, None, :]        # (N, T, S)
    #projected_test = np.einsum('nts,sk->ntk', X_test, eigvecs[:, :NCOMPONENTS])  # (N, T, k)
    #torch.save(torch.from_numpy(eigvecs.copy()), f'pca_eigvecs_meanC{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')

    #X_pred = trf.predict(X_test.reshape(-1, X_test.shape[-1]))

    trf = Ridge(alpha=alpha,fit_intercept = False)
    #pcatrain = np.einsum('ts,sk->tk', X_train, eigvecs[:, :NCOMPONENTS])
    trf.fit(X_train, X_pca)

    if AVG_IND:
        torch.save(torch.from_numpy(eigvecs.copy()).to(torch.float64),f'pca_eigvecs_meanC{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')
        torch.save(torch.from_numpy(eigvals.copy()).to(torch.float64),f'pca_eigval_meanC{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')
    else:
        torch.save(torch.from_numpy(eigvecs.copy()).to(torch.float64),f'pca_eigvecs_meanC{cutoff}{maxang}{maxrad}_sig{0}.pt')
        torch.save(torch.from_numpy(eigvals.copy()).to(torch.float64),f'pca_eigval_meanC{cutoff}{maxang}{maxrad}_sig{0}.pt')
    #X_test_centered = X_test - mu_global[None, None, :]        # (N, T, S)
    #projected_test = np.einsum('nts,sk->ntk', X_test, eigvecs[:, :NCOMPONENTS])  # (N, T, k)
    #torch.save(torch.from_numpy(eigvecs.copy()), 'pca_eigvecs_meanC{cutoff}{maxang}{maxrad}_sig{SIGMA}.pt')

    #X_pred = trf.predict(X_test.reshape(-1, X_test.shape[-1]))
    X_pred = pca.transform(X_test.reshape(-1, X_test.shape[-1]))
    
    return eigvals, eigvecs, X_pred.reshape(Ntest, Ttest, -1), covariances


def pca_model_all_meanC(XA, XB, X_test, NCOMPONENTS=4):
    
    N, T, S = XA.shape
    Ntest, Ttest, Stest = X_test.shape
    eps = 1e-10

    # 1. Concatenate along time axis → (N, 2T, S)
    X_train = np.concatenate([XA, XB], axis=1)
    X_train = X_train.reshape(-1, X_train.shape[-1])
    # 2. Per-atom mean: shape (S)
    mu = X_train.mean(axis=0)

    # 3. Centered data
    X_centered = X_train - mu# (2T*N, S)

    # 4. Per-atom covariance matrices using einsum
    covariances = np.einsum('ts,tk->ks', X_centered, X_centered) / (2*T*N - 1)

    # 5. Average over atoms
    #avg_cov = covariances.mean(axis=0)

    # 6. Eigenvalue decomposition of covariance matrix
    
    
    eigvals, eigvecs = eigh(covariances + eps * np.eye(S))  # Regularize

    # Sort eigenvectors by descending eigenvalue
    eigvals = np.array(eigvals)[::-1]
    eigvecs = np.array(eigvecs)[:, ::-1]

    # 7. Project centered test data: center using mean of train data
    X_test_centered = X_test.reshape(-1, X_test.shape[-1]) - mu
    projected_test = np.einsum('ts,sk->tk', X_test_centered, eigvecs[:, :NCOMPONENTS])

    return eigvals, eigvecs, projected_test.reshape(Ntest,Ttest,-1), covariances



def ridge_model_all(pcaA, pcaE, XA, XE, XA_trf_test, XE_trf_test, XA_clf_test,
                    XE_clf_test, ids_atoms, nstepA, nstepE, alpha, C):
    models = []
    # pca C,N,S
    # all else also C,N,S
    #pcA = pcaA.reshape(nstepA, len(ids_atoms),
    #                   pcaA.shape[-1]).transpose(1,0,2)
    #pcE = pcaE.reshape(nstepE, len(ids_atoms),
     #                  pcaE.shape[-1]).transpose(1,0,2)
    
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

    trf = Ridge(alpha=alpha,fit_intercept = False)
    trf.fit(np.hstack(XA), pcaA)
    trf.fit(np.hstack(XE), pcaE)
    X_clf_train = trf.predict(np.hstack(X_trf_test))
    X_final_test = trf.predict(np.hstack(X_clf_test))

    y_clf_train = y_trf_test
    # train model with those train predictions and eval on test predictions
    # Standardize features
    scaler = StandardScaler()
    print('X_clf_train', X_clf_train.shape)
    print('X_clf_test', X_final_test.shape)
    X_clf_train_scaled = scaler.fit_transform(X_clf_train)
    X_final_test_scaled = scaler.transform(X_final_test)

    # Logistic Regression with L1 (Lasso) penalty
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42, fit_intercept=True)
    model.fit(X_clf_train_scaled, y_clf_train)
    y_pred = model.predict(X_final_test_scaled)
    CV_proba = model.predict_proba(X_final_test_scaled)
    
    print('Precision: TP/(TP + FP)')
    print('Recall: TP/P')
    print(classification_report(y_clf_test, y_pred))
    ll = brier_score_loss(y_clf_test, CV_proba[:,1])
    print('ypred', y_pred)
    print('ytrue', y_clf_test)
    print('proba', CV_proba)
    print('brier loss: ', ll)
    coef = model.coef_[0]
    coef_i = np.where(abs(coef) > 0.001)[0]
    #print(coef_i)

    return X_final_test, y_pred, y_clf_test, np.round(ll,4)

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

    return np.array(timeseries) # C, N, S

def get_SOAPPS_new(traj, ids_atoms, calculator):
    atomselection=Labels_torch(
        names=["atom"],
        values=torch.arange(len(traj[0]), dtype=torch.int64).unsqueeze(-1),
    )
    systems = systems_to_torch(traj, dtype=torch.float64)
    soap = calculator(systems,
            selected_samples=atomselection,selected_keys=selection_torch)
    soap = soap.keys_to_samples("center_type")
    #soap = soap.keys_to_properties(soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"]))
    soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    soap_block = soap.block()
    soap_values = soap_block.values
    
    #np.savetxt('SOAP_values.txt', soap_block.values.reshape(256, -1))
    #np.savetxt('SOAP_values.txt', soap_block.values.squeeze(1))
    #np.savetxt('SOAP_samples.txt', soap_block.samples.values)
    #np.savetxt('SOAP_props.txt', soap_block.properties.values)
    #np.savetxt('pos', traj[0].positions)
    return soap_values.numpy().reshape(-1,len(ids_atoms),soap_values.shape[-1]).transpose(1,0,2)


def get_gaussian_avg(timeseries, ids_atoms):
    gaussian_average = []
    for i in ids_atoms:
        gaussian_average.append(scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
            timeseries[i], sigma=5, axis=0, mode='nearest'))#'reflect')
    return np.array(gaussian_average) # C, N, S

def get_pca(soaps, ids_atoms, pca):
    #print('soaps', soaps.shape)
    print('hstack', np.hstack([soaps[i] for i in ids_atoms]).shape)
    struct_soap_pca = pca.transform(
    np.hstack([soaps[i] for i in
              ids_atoms])) # from C, N, S to NC, S
    print('pcaoutstruct', struct_soap_pca.shape)
    return struct_soap_pca

def get_pca_direct(soaps, ids_atoms, pca):
    X = np.hstack([soaps[i] for i in ids_atoms]) - pca.mean_
    struct_soap_pca = X @ pca.components_.T  # Project onto PCA components
    return struct_soap_pca

def croptrj(trj):
    trj_out = []
    for atoms in trj:
        trj_out.append(atoms[:196])
    return trj_out

#cyclo
selection = Labels(
    names=["center_type", "neighbor_1_type", "neighbor_2_type"],
    #values=np.array([[8,1,1],[8,8,1],[8,8,8]], dtype=np.int32),
    values=np.array([[i,j,k] for i in [6] for j in [1,6,7,8] for k in [1,6,7,8] if j <= k], dtype=np.int32),
)
#waterice
selection = Labels(
    names=["center_type", "neighbor_1_type", "neighbor_2_type"],
    #values=np.array([[8,1,1],[8,8,1],[8,8,8]], dtype=np.int32),
    values=np.array([[i,j,k] for i in [8] for j in [1,8] for k in [1,8] if j <= k], dtype=np.int32),
)
selection_torch = Labels_torch(
    names=["center_type", "neighbor_1_type", "neighbor_2_type"],
    #values=np.array([[8,1,1],[8,8,1],[8,8,8]], dtype=np.int32),
    values=torch.tensor([[i,j,k] for i in [8] for j in [1,8] for k in [1,8] if j <=
        k], dtype=torch.int32),
)



# load trj only once 
#trajA = aseio.read('prodA.pdb', index='::200')[:]
#print('done with A')
#trajE = aseio.read('prodE.pdb', index='::200')[:]
#cyclo
#trajA_mol = aseio.read('prodA-hf.pdb', index='::10')[:]
#print('done with A')
#trajE_mol = aseio.read('prodE-hf.pdb', index='::10')[:]
#print('done with E')
# water
trajA_mol = aseio.read('/Users/markusfasching/EPFL/Work/project1/scripts/SOAP-CV/traj_water_crop.xyz', index='::1')[:]
#trajA_mol = trajA_mol[:int(0.5*len(trajA_mol))]
print('done with A', len(trajA_mol))
trajE_mol = aseio.read('/Users/markusfasching/EPFL/Work/project1/scripts/SOAP-CV/traj_ice_crop.xyz', index='::1')[:]
#trajE_mol = trajE_mol[:int(0.5*len(trajA_mol))]
print('done with E', len(trajE_mol))
trajtest = aseio.read('/Users/markusfasching/EPFL/Work/project1/scripts/SOAP-CV/traj_ice_crop.xyz', index=':1')
#trajtest = trajtest[:int(len(trajtest)*0.5)]
len(trajtest)
print('done', len(trajtest))
#trajA_mol = croptrj(trajA)
print('cropped A')
#trajE_mol = croptrj(trajE)
print('cropped E')
#aseio.write('prodA-mol.pdb', trajA_mol)
#aseio.write('prodE-mol.pdb', trajE_mol)

NA = len(trajA_mol)
NE = len(trajE_mol)
atoms = trajA_mol[0]
# use only C atoms
ids_atoms = [atom.index for atom in atoms if atom.symbol == 'C']
# use a fifth of all oxygens for water and ice!
ids_atoms = [atom.index for atom in [atom for atom in atoms if atom.symbol =='O'][::1]]
atoms = trajtest[0]
ids_atoms_test = [atom.index for atom in [atom for atom in atoms if atom.symbol =='O']]
print('len ids_atoms', len(ids_atoms))
print('len ids_Atoms_test', len(ids_atoms_test))
print('test length', len(atoms))
from sklearn.preprocessing import StandardScaler
from copy import deepcopy 
for cutoff in [5.0]: #[0.5, 0.1, 0.01]:
    for maxang in [6]: #[0.1, 0.05, 0.01]:
        for maxrad in [6]:
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
                    "max_angular": maxang, #8
                    "radial": {"type": "Gto", "max_radial": maxrad}, #6
                },
            }

            calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
            
            trajA_copy = [atoms.copy() for atoms in trajA_mol]  # safe ASE Atoms copy
            trajE_copy = [atoms.copy() for atoms in trajE_mol]  # safe ASE Atoms copy
            trajtest_copy = [atoms.copy() for atoms in trajtest]  # safe ASE Atoms copy
            ids_copy = ids_atoms.copy()  # if this is a list or ndarray

            timeseries_test = get_SOAPPS_new(trajtest_copy, ids_atoms_test, calculator)
            print('newtsshapetest', timeseries_test.shape)
            #np.savetxt('SOAP0.txt', timeseries_test.reshape(256, -1)) 
            timeseriesA_train = get_SOAPPS_new(trajA_copy, ids_copy, calculator)

            timeseriesE_train = get_SOAPPS_new(trajE_copy, ids_copy, calculator)

            #timeseries_test = get_SOAPPS(trajtest_copy, ids_atoms_test, calculator)
            #timeseries_test = get_SOAPPS_new(trajtest_copy, calculator)
            #print('newtsshapetest', timeseries_test.shape)
            nstepA = timeseriesA_train.shape[1]
            nstepE = timeseriesE_train.shape[1]
            y = np.concatenate((np.ones(nstepA), np.zeros(nstepE)))
            ids_atoms_i = np.arange(0,len(ids_atoms))
            X = np.concatenate((timeseriesA_train, timeseriesE_train), axis=1)

            #gaussian_averageA_train = get_gaussian_avg(timeseriesA_train, ids_atoms_i) #CN, S
            #print('gaussianaverage', gaussian_averageA_train.shape)

            #gaussian_averageE_train = get_gaussian_avg(timeseriesE_train, ids_atoms_i)

            
            #X_test = np.concatenate((timeseriesA_test, timeseriesE_test), axis=1)
            X_test = timeseries_test
            interface = 20
            buffer = 5
            #oxygen_indices = [atom.index for atom in timeseries_test.shape[0] if atom.symbol == 'O']
            y_test = np.full((len(trajtest_copy), len(ids_atoms_test)), 2)
            #print('len trajtest', len(trajtest))
            #print('Xtest', X_test.shape)
            #print('ytest', y_test.shape)
            for nframes, atoms in enumerate(trajtest_copy):
                oxygens = atoms[ids_atoms_test]
                pos = oxygens.get_positions()
                icetest = pos[:,2] < interface - buffer
                watertest = pos[:,2] > interface + buffer
                #interfacetest = (interface - buffer < pos[:2]) & (pos[:2] < interface + buffer)
                #print('posshape', pos[:,2].shape)
                #print('ytest[frames]', y_test[nframes].shape)
                y_test[nframes][watertest] = 0 # WATER --> 0
                y_test[nframes][icetest] = 1 # ICE --> 1
                #y_test[nframes][interfacetest] = 2 # INTERFACE --> 2

            for SIGMA in [5, 200, 25, 100]:               
                
                eigvals_lda, eigvecs_lda, projected_test_lda, S_B_lda, S_W_lda = lda_model_all_meanC_ridge(
                                                    timeseriesA_train, 
                                                    timeseriesE_train, 
                                                    timeseries_test,
                                                    AVG_IND=False,
                                                    SIGMA=SIGMA,
                                                    )
               
                eigvals_ldaga, eigvecs_ldaga, projected_test_ldaga, S_B_ldaga,S_W_ldaga = lda_model_all_meanC_ridge(
                                                    timeseriesA_train, 
                                                    timeseriesE_train, 
                                                    timeseries_test,
                                                    AVG_IND=True,
                                                    SIGMA=SIGMA,
                                                    )

                
                eigvals_pca_avg, eigvecs_pca_avg, projected_test_pca_avg, cov_pca_avg = pca_model_all_meanC_ridge(
                    timeseriesA_train, 
                    timeseriesE_train, 
                    timeseries_test,
                    AVG_IND=True,
                    SIGMA=SIGMA,
                    )
                
                eigvals_pca, eigvecs_pca, projected_test_pca, cov_pca = pca_model_all_meanC_ridge(
                    timeseriesA_train, 
                    timeseriesE_train, 
                    timeseries_test,
                    AVG_IND=False,
                    SIGMA=SIGMA,
                    )
                
                

               
                X_test_proj = np.array([projected_test_lda, projected_test_ldaga, projected_test_pca, projected_test_pca_avg])
                #X_test_proj = X_test_proj.transpose(0,2,1,3).reshape(4, -1, X_test_proj.shape[-1]) #swap atom and time axis
                X_test_proj = X_test_proj.transpose(0,2,1,3).reshape(4, -1, X_test_proj.shape[-1]) #swap atom and time axis to model, T, C, S
                #y_test = np.concatenate((np.ones(timeseriesA_test.shape[1]), np.zeros(timeseriesE_test.shape[1])))
                y_test = y_test.reshape(-1) #flatten like we flatten X_test
                """
                print('avg2avg')
                print('gaussianaverage', gaussian_averageA.shape)
                Xridge_avg2avg, yridge_avg2avg, yridge_avg2avg_true, ll_avg2avg = ridge_model_all(pcaA, pcaE, gaussian_averageA, gaussian_averageE,
                                  timeseriesA_test_trf, timeseriesE_test_trf,
                                  timeseriesA_test_clf, timeseriesE_test_clf, ids_atoms_i,
                                  nstepA, nstepE, alpha, C)
                print('avg2raw')
                Xridge_avg2raw, yridge_avg2raw, yridge_avg2raw_true, ll_avg2raw = ridge_model_all(pcaA_raw, pcaE_raw, gaussian_averageA, gaussian_averageE,
                                  timeseriesA_test_trf, timeseriesE_test_trf,
                                  timeseriesA_test_clf, timeseriesE_test_clf, ids_atoms_i,
                                  nstepA, nstepE, alpha, C)
                print('raw2raw')
                Xridge_raw2raw, yridge_raw2raw, yridge_raw2raw_true, ll_raw2raw = ridge_model_all(pcaA_raw,
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
                                                                                      alpha, 
                                                                                      C,
                                                                                      )
                print('raw2avg')
                Xridge_raw2avg, yridge_raw2avg, yridge_raw2avg_true, ll_raw2avg = ridge_model_all(pcaA,
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
                                                                                      alpha,
                                                                                      C,
                                                                                      )
                """
                #cwdA = '/Users/markusfasching/EPFL/Work/project-CVs/cyclosporin/gromacs/cycloA/'
                #uA = mda.Universe(os.path.join(cwdA, "prod.tpr"), os.path.join(cwdA, "prod.xtc"))
                #Cs = uA.select_atoms('element C')
                #feature_names = [f'{C.name} from {C.resname} PCA {i+1}' for C in Cs for i in range(3)]
                #feature_names_C = [f'{C.name} from {C.resname}' for C in Cs]
                fig, ax = plt.subplots(2, 4, figsize=(13,7))
                for i in range(2):
                    ax[i,0].set_title(f'{i+1} lda')
                    ax[i,1].set_title(f'{i+1} lda time avg')
                    ax[i,2].set_title(f'{i+1} pca')
                    ax[i,3].set_title(f'{i+1} pca time avg')
                    
                    
                    #if i in coef_:
                    #    ax[i,0].set_title(f'{feature_names_C[i]} PCA from avg', color='red')
                    ##else:
                    #    ax[i,0].set_title(f'{feature_names_C[i]} PCA from avg')
                    pca1_i = i*2
                    pca2_i = pca1_i+1
                    
                    #maxs = np.max(np.abs(X_test_proj), axis=(0,1))
                    #mins = -maxs
                    label = ['water', 'ice', 'interface']
                    for n, testdata in enumerate(X_test_proj):
                        #cyclo
                        # ax[i,n].scatter(testdata[:,pca1_i], testdata[:,pca2_i], c=y_test, cmap='coolwarm', alpha=0.5) #C=50
                        ax[i,n].scatter(testdata[:,pca1_i],
                                testdata[:,pca2_i],s=1, c=y_test, cmap='viridis', alpha=0.05) #C=50
                        ax[i,n].legend()
                    """for j in range(4):
                        ax[i,j].set_xlim(mins[pca1_i], maxs[pca1_i])
                        ax[i,j].set_ylim(mins[pca2_i], maxs[pca2_i])
                       """
                    
                    #ax[i,2].scatter(Xridge_raw_scaled[:,pca1_i], Xridge_raw_scaled[:,pca2_i], c=y_raw_scaled, cmap='coolwarm', alpha=0.5, label='scaled raw') #C=50
                    #ax[i,3].scatter(Xridge_raw[:,pca1_i], Xridge_raw[:,pca2_i], c=y_raw, cmap='coolwarm', alpha=0.5, label='raw') #C=50

                plt.savefig(f'lda-pca-comparison-niu-Cmean_{cutoff}{maxang}{maxrad}_sig{SIGMA}.png')

