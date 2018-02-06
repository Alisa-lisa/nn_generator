from model_generator.simple_nn import SET_UP_KEYS
import json
import yaml

def read_out_json_config(config_name):
    """ As you could guess, reads config from file"""
    with open(config_name, 'r') as config:
        try:
            res = {}
            config_raw = json.load(config)
            if not set(SET_UP_KEYS).issubset(config_raw.keys()):
                raise ValueError("Configuration is incomplete")
            else:
                for k, v in config_raw.items():
                    if k=="architecture":
                        res[k] = {}
                        for k1, v1 in v.items():
                            res[k].update({int(k1):int(v1)})
                    else:
                        res[k] = v
                return res
        except ValueError:
            raise ValueError("Could not read the config file")

def read_out_yaml_config(config_name):
    """ Aaaand again, just reading out the  config"""
    with open(config_name, 'r') as config:
        try:
            config_raw = yaml.load(config)
            if not set(SET_UP_KEYS).issubset(config_raw.keys()):
                raise ValueError("Configuration is incomplete")
            else:
                return config_raw
        except ValueError:
            raise ValueError("Could not read the config file")

def read_out_config(config_name):
    """ Wrapper for both YAML and JSON formats """
    extension = config_name.split(".")[-1]
    if extension == "json":
        try:
            return read_out_json_config(config_name)
        except ValueError:
            raise ValueError("Inappropriate config provided")
    elif extension in ["yml", "yaml"]:
        try:
            return read_out_yaml_config(config_name)
        except ValueError:
            raise ValueError("Inappropriate config provided")
    else:
        raise ValueError("Unknown config file extension")



