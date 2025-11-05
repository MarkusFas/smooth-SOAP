import matplotlib.pyplot as plt
import mpltex
import numpy as np


@mpltex.acs_decorator
def plot_2pca(X, label): #T,N,P
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    for i, trj in enumerate(X.transpose(1,0,2)): #N, T, P
        color = np.arange(len(trj[:,0]))
        sc = ax.scatter(
            trj[:,0], 
            trj[:,1], #first PCA component
            c=color, 
            cmap='RdYlBu', 
            vmin=0, #np.min(color), 
            vmax=len(trj[:,0]), #np.max(color),
            alpha=0.2,
            s=2.5,
        ) 
        
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('z')
    plt.savefig(label + f'_projection.png', dpi=200)


@mpltex.acs_decorator
def plot_2pca_atoms(X, label, sel_atoms):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    for i, trj in enumerate(X.transpose(1,0,2)): # shape N, T, P
        color = 'green' if sel_atoms[i] > 256*3//2 else 'orange'
        sc = ax.scatter(
            trj[:,0], 
            trj[:,1], #first PCA component
            color=color, 
            alpha=0.2,
            s=2.5,
        )
    plt.savefig(label + f'_projection_atoms.png', dpi=200)

@mpltex.acs_decorator
def plot_2pca_height(X, label, sel_atoms, traj):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    positions = np.array([atoms.positions[sel_atoms, 2] for atoms in traj])
    for i, trj in enumerate(X.transpose(1,0,2)): # shape N, T, P
        color = 'green' if sel_atoms[i] > 256*3//2 else 'orange'
        ax.scatter(
            trj[:,0], 
            positions[:,i], #first PCA component
            color=color, 
            alpha=0.2,
            s=2.5,
        )
    plt.savefig(label + f'_projection_height.png', dpi=200)

