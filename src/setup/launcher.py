# src/launcher/run_system.py
from src.setup.defaults import DEFAULT_PARAMS
from src.setup.input_check import setup_simulation
import yaml
import sys

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_params(defaults, updates, path="root"):
    """
    Update default params, check if all used keywords are in the defaults
    """
    for key, value in updates.items():
        if key not in defaults:
            raise KeyError(f"Unknown config key '{key}' at path '{path}'")
        if isinstance(value, dict) and isinstance(defaults[key], dict):
            merge_params(defaults[key], value, path=f"{path}.{key}")
        else:
            defaults[key] = value
    return defaults


if __name__ == "__main__":
    #input_file = 'systems/icewater/test_interval1.yaml'
    #input_file = 'systems/cycloAE/test_sep_tests.yaml'
    #input_file = 'systems/cycloAE/test_intervalpca.yaml'
    #input_file = 'systems/ice_water_sep/test_interval1.yaml'
    #input_file = 'systems/icewater/test_interval1.yaml'
    
    #input_file = 'systems/smallcell_interface_350/test_intervaltemp.yaml'
    #input_file = 'systems/smallcell_interface_350/test_intervalpcata.yaml'
    #input_file = 'systems/test_hannah/test_interval1.yaml'
    #input_file = 'systems/smallcell_interface_350/test_intervaltica.yaml'
    if len(sys.argv)>0:
        input_file=sys.argv[1]
#    input_file = 'systems/icewater/test.yaml'
    
    #input_file = 'systems/ice_water_sep/test_intervaltemp.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)

    print(params)
    print(params['interval'])

#    #for interval in test_intervals:
#    X, properties = SOAP_full(trj, params['interval'], params['train_selected_atoms'], params['SOAP_params'], centers, neighbors, sigma)
#    X_values.append(X[0]) # first center type TxNxD
#
#    for i in range(X_values[0].shape[-1]//20):
#        SOAP_idx = np.arange(i*20, (i+1)*20, 1)
#        #SOAP_idx = np.random.randint(0, X_values[0].shape[-1]//5, 25)
#
#        print(f'done with calculation for {20*i} - {20*(i+1)}')
#        label_used = label + f'_{i*20}-{20*(i+1)}'
#        #plot_compare_spatialave(X_values, SOAP_idx, label_used, properties.values.numpy(), test_sigmas)
#        #plot_compare_atoms_spat(X_values, SOAP_idx, label_used, properties.values.numpy(), test_sigmas)
#        plot_compare_timeave(X_values, SOAP_idx, label_used, properties.values.numpy(), test_intervals)
#        plot_compare_atoms(X_values, SOAP_idx, label_used, properties.values.numpy(), test_intervals)

    exit()
    input_file = 'systems/cycloAE/test_interval1.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)
    exit()
    input_file = 'systems/icewater/test_interval250.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)
