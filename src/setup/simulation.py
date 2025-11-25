import os
import random
import metatomic.torch as mta
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import chain
import chemiscope
from src.plots.cov_heatmap import plot_heatmap
from src.plots.timeseries import plot_projection_atoms, plot_projection_atoms_models
from src.plots.histograms import plot_2pca, plot_2pca_atoms, plot_2pca_height
from src.classifier.Logreg import run_logistic_regression

def run_simulation(trj, trj_test, methods_intervals, **kwargs):
    if trj_test is None:
        trj_test = trj

    if not isinstance(trj[0], list):
        trj = [trj]

    if not isinstance(trj_test[0], list):
        trj_test = [trj_test]

    for i, methods in tqdm(enumerate(methods_intervals), desc="looping through intervals"):
        for j, method in tqdm(enumerate(methods), desc="looping through methods"):
            random.seed(7)
            # create labels and directories for results
            
            N_train = kwargs.get('train_selected_atoms')
            N_test = kwargs.get('test_selected_atoms')
            is_shuffled = False
            if isinstance(N_train , int):
                selected_atoms = [idx for idx, number in enumerate(trj[0][0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
                random.shuffle(selected_atoms) 
                train_atoms = selected_atoms[:N_train]
                is_shuffled = True
            else:
                train_atoms = N_train
            if isinstance(N_test , int):
                if not is_shuffled:
                    selected_atoms = [idx for idx, number in enumerate(trj[0][0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
                    random.shuffle(selected_atoms)
                    test_atoms = selected_atoms[:N_test]
                else:
                    print('test from shuffled')
                    test_atoms = selected_atoms[10+N_train: 10+N_train + N_test]
                    test_atoms = selected_atoms[-N_test:] # single atom case
            else:
                test_atoms = N_test
            if test_atoms is None:
                test_atoms = train_atoms
            print('Ntrain, Ntest: ', N_train, N_test)
            print('Train atoms: {}'.format(train_atoms))        
            print('Test atoms: {}'.format(test_atoms))        
            #train_atoms = selected_atoms
            train_atoms = sorted(train_atoms)
            test_atoms = sorted(test_atoms)
            #print("WARNING SAME TEST AND TRAIN ATOMS")
            #test_atoms = train_atoms
            # train our method by specifying the selected atoms
            method.train(trj, train_atoms)

            method.log_metrics()


            # get predictions with the new representation
            # for prediction we can use the concatenated trajectories


            trj_predict = list(chain(*trj_test))
            X = method.predict(trj_predict, test_atoms) ##centers N,T,P
            X = [proj.transpose(1,0,2) for proj in X]#centers T,N,P

            if kwargs["ridge"]:
                #method.fit_ridge(trj_predict)
                print('fit ridge 1')
  
                #print('fit ridge 2')
                trj_ridge = list(chain(*trj))
                method.fit_ridge_nonincremental(trj_ridge)

                #X_ridge = method.predict_ridge(trj[0], train_atoms)
                X_ridge = method.predict_ridge(trj_predict, test_atoms)
                X_ridge = [proj.transpose(1,0,2) for proj in X_ridge]
            
            if kwargs["output_per_structure"]:
                X = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X]

            # label the trajectories:
            if kwargs['classify']['request']:
                if kwargs['classify']['switch_index'] is not None:
                    y = method.get_label(trj[0], test_atoms, kwargs['classify']['switch_index'])
                elif len(trj) > 1:
                    y = np.concatenate([np.full((len(t),len(test_atoms)), i) for i, t in enumerate(trj)])
                else:
                    ValueError("No labels provided for the trajectory. Please provide 'switch_index' or multiple trajectories for classification.")

                # Classificatoin
                run_logistic_regression(
                    X[0], y, 
                    outfile_prefix=method.label + '_logreg',
                    random_state=42,
                    solver='lbfgs',
                    max_iter=500
                )
            
            
            #4 Post processing

            plots = kwargs.get("plots", [])

            
            if "projection" in plots:
                plot_projection_atoms(X, [0,1,2,3], method.label, [method.interval]) # need to transpose to T,N,P
                if kwargs["ridge"]:
                    plot_projection_atoms(X_ridge, [0,1,2,3], method.label + '_ridge', [method.interval]) # need to transpose to T,N,P
                #plot_projection_atoms_models(X, [0,1,2,3], label, [method.interval])
                print('Plotted projected timeseries for test atoms')

            if "pca" in plots:
                for i, proj in enumerate(X):
                    plot_2pca(proj, method.label + f'_{i}')
                if kwargs["ridge"]:
                    for i, proj in enumerate(X_ridge):
                        plot_2pca(proj, method.label + f'_ridge_{i}')
                print('Plotted scatterplot of PCA')

            if "pca_atoms" in plots:
                for i, proj in enumerate(X):
                    plot_2pca_atoms(proj, method.label + f'_{i}', test_atoms)
                if kwargs["ridge"]:
                    for i, proj in enumerate(X_ridge):
                        plot_2pca_atoms(proj, method.label + f'_ridge_{i}', test_atoms)
                print('Plotted scatterplot of PCA atoms labels')
            
            if "pca_height" in plots:
                for i, proj in enumerate(X):
                    plot_2pca_height(proj, method.label + f'_{i}', test_atoms, trj_predict)
                if kwargs["ridge"]:
                    for i, proj in enumerate(X_ridge):
                        plot_2pca_height(proj, method.label + f'_ridge_{i}', test_atoms, trj_predict)
                print('Plotted scatterplot of PCA height labels')

            print('Plots saved at ' + method.label)

            if "cs" in plots:
                cs = chemiscope.show(trj[0],
                    properties={
                        "PC[0]": {"target": "atom", "values": X[0][...,0].flatten()},
                        "PC[1]": {"target": "atom", "values": X[0][...,1].flatten()},
                        "time": {"target": "atom", "values": np.repeat(np.arange(X[0].shape[0]), X[0].shape[1])},
                    },
                    environments = [[i,j,4] for i in range(X[0].shape[0]) for j in test_atoms], # maybe range(X[0].shape[1])
                    settings=chemiscope.quick_settings(periodic=True, trajectory=True, target="atom", map_settings={"joinPoints": False})
                )
                cs.save(method.label + '_cs.json')
                print("saved chemiscope")

            if kwargs["model_save"]:
                for i, trans in enumerate(method.transformations):
                    method.descriptor.set_atom_types(trj)
                    if kwargs["ridge"] and kwargs["ridge_save"]:
                        method.descriptor.set_projection_matrix(method.ridge[i].coef_.T)
                    else:
                        method.descriptor.set_projection_matrix(trans.eigvecs)
                    method.descriptor.set_projection_dims(dims=kwargs['model_proj_dims'])
                    method.descriptor.set_projection_mu(mu=trans.mu)
                    method.descriptor.eval()   
                    #method.descriptor.save_model(path=method.root+f'/interval_{method.interval}/', name='model_soap')   
                    #print(f'saved model at {method.root}'+f'/interval_{method.interval}/')    
                    method.descriptor.save_model(path=method.label, name='model_soap') 


                    
    if ("heatmap" in plots) and len(methods_intervals) >= 2:
        interval_0 = methods_intervals[0]
        interval_1 = methods_intervals[1]
        cov1_int0 = interval_0[0].cov_mu_t
        cov2_int0 = interval_0[0].mean_cov_t
        cov1_int1 = interval_1[0].cov_mu_t
        cov2_int1 = interval_1[0].mean_cov_t
        for i, center in enumerate(interval_0[0].descriptor.centers):
            plot_heatmap(cov1_int0[i], cov1_int1[i], method.root + f'_temporal_interval{interval_0[0].interval}{interval_1[0].interval}_center{center}' + f'_{i}')
            plot_heatmap(cov2_int0[i], cov2_int1[i], method.root + f'_spatial_interval{interval_0[0].interval}{interval_1[0].interval}_center{center}' + f'_{i}')
        print('Plotted heatmap')
 
    
if __name__ == '__main__':
    print('Nothing to do here')
