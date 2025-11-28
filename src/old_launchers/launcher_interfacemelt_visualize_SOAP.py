from src.setup.read_data import read_trj, tamper_water_trj
from src.old_scripts.descriptors import SOAP_mean, SOAP_full
from src.transformations.PCAtransform import pcatransform, PCA_obj

from src.old_scripts.fourier import fourier_trafo
from src.visualize import plot_compare_atoms, plot_compare_timeave, plot_compare_spatialave, plot_compare_atoms_spat

import numpy as np
from tqdm import tqdm
import chemiscope
from pathlib import Path
import os
import random


#data1 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/icemeltinterface/TIP4P/positions.extxyz'
#data1 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/interfaces/250_275/positions.lammpstrj'
data1 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/interfaces/250_275_fast/positions.lammpstrj'
#data1 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/interfaces/250/positions.lammpstrj'
SOAP_cutoff = 5
SOAP_max_angular = 6
SOAP_max_radial = 6

centers = [8] # center on Te
neighbors  = [1,8]

HYPER_PARAMETERS = {
    "cutoff": {
        "radius": SOAP_cutoff, #4 #5 #6
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.25, #changed from 0.3
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": SOAP_max_angular, #8
        "radial": {"type": "Gto", "max_radial": SOAP_max_radial}, #6
    },
}


if __name__=='__main__':
    trj1 = read_trj(data1, ':')
    #trj = trj1[:1000] + trj1[-1000:]
    trj = trj1[::5]
    #trj = tamper_water_trj(trj)
    #trj = trj[:2]

    print('done loading the structures')
    dir = f'results/icewaterinterfacemeltfast_lf/ICESPATIAL/250_275_fast/test/visual_constantavg_stride/CUTOFF/SOAP_deeptime_single'
    #dir = f'results/icewaterinterfacemelt/ALLSPATIAL/visual_constantavg_stride/CUTOFF/SOAP_deeptime_single'
    Path(dir).mkdir(parents=True, exist_ok=True)
    
    label = f'SOAP_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
    label = os.path.join(dir, label)
    
    ids_atoms_train = [atom.index for atom in trj[0] if atom.number == centers[0]]
    #ids_atoms_train = ids_atoms_train[2*len(ids_atoms_train)//3:]
    random.seed(7)
    random.shuffle(ids_atoms_train)
    ids_atoms_train = ids_atoms_train[:30]
    ids_atoms_train = [705, 522, 699, 696, 693, 690, 513, 684, 540, 678, 570, 672, 669, 666, 591, 600]

    test_intervals = [1, 10, 100, 250]
    X_values = []
    #for interval in test_intervals:
    interval =1 
    sigma=0
    test_sigmas = [1,3,5]
    for interval in test_intervals:
        X, properties = SOAP_full(trj, interval, ids_atoms_train, HYPER_PARAMETERS, centers, neighbors, sigma)
        X_values.append(X[0]) # first center type TxNxD

    """SOAP_idx = 10
    for SOAP_idx in range(12):
        for i, SOAPS in enumerate(X_values):
            SOAP_placeholder = np.zeros((SOAPS.shape[0], len(trj[0]), 1))

            SOAP_placeholder[:,ids_atoms_train,0] = SOAPS[...,SOAP_idx]
            fname = dir + f'/SOAP_interval{test_intervals[i]}_sigma{test_sigmas[0]}_SOAP_{SOAP_idx}'
            cs = chemiscope.show(trj,
                properties={
                    "SOAP": {"target": "atom", "values": SOAPS[...,SOAP_idx].flatten()},
                    #"time": np.arange(SOAPS.shape[0]),
                    "time": {"target": "atom", "values":np.repeat(np.arange(SOAPS.shape[0]), SOAPS.shape[1])},
                },
                environments = [[i,j,4] for i in range(SOAPS.shape[0]) for j in ids_atoms_train], # maybe range(X[0].shape[1])
                settings=chemiscope.quick_settings(periodic=True, trajectory=True, target="atom", map_settings={"joinPoints": False})
            )
            cs.save(fname + '_cs.json')
    print("saved chemiscope")"""
    twothird = 2*(X_values[0].shape[-1]//20)//3
    for i in np.arange(X_values[0].shape[-1]//20)[twothird:twothird+5]:
        SOAP_idx = np.arange(i*20, (i+1)*20, 1)
        #SOAP_idx = np.random.randint(0, X_values[0].shape[-1]//5, 25)

        print(f'done with calculation for {20*i} - {20*(i+1)}')
        label_used = label + f'_{i*20}-{20*(i+1)}'
        #plot_compare_spatialave(X_values, SOAP_idx, label_used, properties.values.numpy(), test_sigmas)
        #plot_compare_atoms_spat(X_values, SOAP_idx, label_used, properties.values.numpy(), test_sigmas)
        plot_compare_timeave(X_values, SOAP_idx, label_used, properties.values.numpy(), test_intervals)
        plot_compare_atoms(X_values, SOAP_idx, label_used, properties.values.numpy(), test_intervals)
