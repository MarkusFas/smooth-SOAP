from tropea_clustering import onion_multi, helpers
import numpy as np
import matplotlib.pyplot as plt


def plot_onion(X, label, sel_atoms, traj):
    # Select time resolution
    delta_t = 1
    # Create input array with the correct shape only for first X dim for now!
    data = X.transpose(2,1,0)[:2,...]
    reshaped_input_data = helpers.reshape_from_dnt(data, delta_t)
    # Run Onion Clustering
    state_list, labels = onion_multi(reshaped_input_data)
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    x = np.linspace(np.min(data), np.max(data), 200)
    for i, state in enumerate(state_list):
        mu = state.mean[0]
        var = state.sigma[0]**2
        y = (1/np.sqrt(2*np.pi*var)) * np.exp(-0.5*((x - mu)**2 / var))
        ax.plot(x, y, label=f'cluster_{i}')
    #add histogram
    T, N, P = X.shape
    data = X.reshape(-1, P)
    ax.hist(data[:,0], bins=50, alpha=0.4, density=True)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig(label + '_onion_states.png', dpi=300)