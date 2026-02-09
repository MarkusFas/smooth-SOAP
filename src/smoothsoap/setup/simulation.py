import os
import random

import metatomic.torch as mta
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import chain
import chemiscope

from smoothsoap.plots.cov_heatmap import plot_heatmap
from smoothsoap.plots.post_processing import post_processing
from smoothsoap.classifier.Logreg import run_logistic_regression

def run_simulation(trj, trj_test, methods_intervals, **kwargs):
    is_shared = False
    if trj_test is None:
        is_shared = True
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
                if is_shared:
                    if not is_shuffled:
                        selected_atoms = [idx for idx, number in enumerate(trj_test[0][0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
                        random.shuffle(selected_atoms)
                    test_atoms = selected_atoms[-N_test:]
                else:
                    print('test from shuffled')
                    selected_atoms = [idx for idx, number in enumerate(trj_test[0][0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
                    random.shuffle(selected_atoms)
                    test_atoms = selected_atoms[-N_test:] # single atom case
            else:
                test_atoms = N_test
            if test_atoms is None:
                test_atoms = train_atoms
            #print('Ntrain, Ntest: ', N_train, N_test)
            #print('Train atoms: {}'.format(train_atoms))        
            #print('Test atoms: {}'.format(test_atoms))        
            #train_atoms = selected_atoms
            train_atoms = sorted(train_atoms)
            test_atoms = sorted(test_atoms)
            #print("WARNING SAME TEST AND TRAIN ATOMS")
            #test_atoms = train_atoms
            # train our method by specifying the selected atoms
            method.train(trj, train_atoms)

            # for saving eigvecs, eigvals, mu etc for analysis
            if kwargs['log']:
                method.log_metrics()


            # get predictions with the new representation
            # for prediction we can use the concatenated trajectories


            trj_predict = list(chain(*trj_test))
            if kwargs["ridge"]:
                #method.fit_ridge(trj_predict)
                print('Starting to fit the Ridge ...')

                #print('fit ridge 2')
                trj_ridge = list(chain(*trj))
                method.fit_ridge_nonincremental(trj_ridge)
                print('Finished the Ridge fit')
                #X_ridge = method.predict_ridge(trj[0], train_atoms)
                print('Starting Ridge prediction ...')
                X_ridge = method.predict_ridge(trj_predict, test_atoms)
                X_ridge = [proj.transpose(1,0,2) for proj in X_ridge]
                print('Finished Ridge prediction')
            
            print('Starting to predict ...')
            X = method.predict(trj_predict, test_atoms) ##centers N,T,P
            print('Finished the prediction')
            X = [proj.transpose(1,0,2) for proj in X]#centers T,N,P

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

            if kwargs["model_save"]:
                for i, trans in enumerate(method.transformations):
                    method.descriptor.set_atom_types(trj)
                    keys_to_save = ['system', 'version', 'specifier',
                                    'train_selected_atoms', 'test_selected_atoms', 'input_params', 
                                    'output_params', 'descriptor', 'SOAP_params', 'ridge', 
                                    'ridge_save', 'model_proj_dims', 'i_pca', 'classify', 'base_path']
                    run_specific={'method':method.name, 'interval':method.interval,
                                  'descriptor': method.descriptor 
                                  }
                    savekwargs={key: kwargs[key] for key in keys_to_save}
                    method.descriptor.update_hypers(savekwargs)
                    method.descriptor.update_hypers(run_specific)

                    if hasattr(method, 'spatial_cutoff'):
                        method.descriptor.update_hypers({'spatial_cutoff':method.spatial_cutoff})
                    if hasattr(method, 'sigma'):
                        method.descriptor.update_hypers({'sigma':method.sigma})
                    if hasattr(method, 'n_cumulants'):
                        method.descriptor.update_hypers({'n_cumulants':method.cumulants})
                    if hasattr(method, 'lag'):
                        method.descriptor.update_hypers({'lag':method.lag, 'max_lag':kwargs['max_lag'], 'min_lag':kwargs['min_lag'],'lag_step':kwargs['lag_step']})
                    if kwargs['ridge']==True:
                        method.descriptor.update_hypers({'ridge_alpha':method.ridge_alpha})


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

            #4 Post processing
            post_processing(X, trj_predict, test_atoms, method, method.label, **kwargs)
            if kwargs["ridge"]:
                post_processing(X_ridge, trj_predict, test_atoms, method, method.label + f'_ridge', **kwargs)
            if kwargs["predict_avg"] and (method.name == "SpatialPCA" or method.name == "PCAfull"):
                X_fromavg = method.predict_avg(trj_predict, test_atoms) ##centers N,T,P
                X_fromavg = [proj.transpose(1,0,2) for proj in X_fromavg]
                print('Finished the prediction for averaged')
                post_processing(X_fromavg, trj_predict, test_atoms, method, method.label + f'_fromavg', **kwargs)
            if kwargs["output_per_structure"]:
                X = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X]
                newlabel = method.label + f"_per_structure"
                post_processing(X, trj_predict, test_atoms, method, newlabel, **kwargs)
                if kwargs["ridge"]:
                    X_ridge = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X_ridge]
                    post_processing(X_ridge, trj_predict, test_atoms, method, newlabel+ f'_ridge', **kwargs)
                if kwargs["predict_avg"] and (method.name == "SpatialPCA" or method.name == "PCAfull"):
                    X_fromavg = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X_fromavg]
                    post_processing(X_fromavg, trj_predict, test_atoms, method, newlabel + f'_fromavg', **kwargs)

if __name__ == '__main__':
    print('Nothing to do here')
