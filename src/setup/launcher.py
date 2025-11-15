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
    try:
        if len(sys.argv)>0:
            input_file=sys.argv[1]
    except IndexError:
        pass
    #input_file = 'systems/icewater/test.yaml'
    #input_file = 'systems/icewater/test_interval1.yaml'
    #input_file = 'systems/cycloAE/test_sep_tests.yaml'
    #input_file = 'systems/cycloAE/test_intervalpca.yaml'
    #input_file = 'systems/ice_water_sep/test_interval1.yaml'
    #input_file = 'systems/smallcell_interface_350/test_interval_lf.yaml'
    #input_file = 'systems/icewater/test_interval1.yaml'

    #input_file = 'systems/smallcell_interface_350/test_intervaltemp.yaml'
    input_file = 'systems/cycloAE/test_interval_hf1.yaml'
    input_file = 'systems/smallcell_interface_350/test_interval_lf0.yaml'
    #input_file = 'systems/ice_water_sep/test_interval1.yaml'
    #input_file = 'systems/GeTe/test_interval1.yaml' 


    #input_file = 'systems/cycloAE/test_interval_hf1.yaml'
    #input_file = 'systems/test_hannah/test_interval1.yaml'
    #input_file = 'systems/smallcell_interface_350/test_intervaltica.yaml'

    #input_file= 'systems/BaTiO3/test.yaml'

    #if len(sys.argv)>0:
    #    input_file=sys.argv[1]
    #else:
    #   print('Please provide default file')

    #input_file = 'systems/ice_water_sep/test_intervaltemp.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)
    exit()
    input_file = 'systems/smallcell_interface_350/test_interval_lf2.yaml'
    #input_file = 'systems/cycloAE/test_interval_hf1.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)
    exit()
    input_file = 'systems/GeTe/test_interval1.yaml'
    #input_file = 'systems/cycloAE/test_interval_hf1.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)
    exit()
    input_file = 'systems/smallcell_interface_350/test_interval1_select_ice.yaml'
    user_cfg = load_config(input_file)
    params = merge_params(DEFAULT_PARAMS, user_cfg, input_file)
    setup_simulation(**params)
