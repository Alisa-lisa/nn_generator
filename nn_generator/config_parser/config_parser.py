import json
import logging

import yaml

from nn_generator.model_generator.simple_nn import MUST_KEYS


def read_out_json_config(config_name):
    """ As you could guess, reads config from file"""
    with open(config_name, 'r') as config:
        try:
            res = {}
            config_raw = json.load(config)
            if not set(MUST_KEYS).issubset(config_raw.keys()):
                raise ValueError("Configuration is incomplete")
            else:
                for k, v in config_raw.items():
                    if k == "architecture":
                        res[k] = {}
                        for k1, v1 in v.items():
                            res[k].update({int(k1): int(v1)})
                    elif k == "activation":
                        res[k] = {}
                        for k1, v1 in v.items():
                            res[k].update({int(k1): str(v1)})
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
            if not set(MUST_KEYS).issubset(config_raw.keys()):
                raise ValueError("Configuration is incomplete")
            else:
                return config_raw
        except ValueError:
            raise ValueError("Could not read the config file")


def is_valid_config(config):
    """
    Checks that all must-keys and their must-values are fine
    also checks might keys and their values
    all "non-registered" keys are ignored
    :param config: dict containigng desired keys and values
    :return: bool
    """
    is_valid = True
    for k, v in config.items():
        if k == "architecture":
            keys = [i for i in range(1, len(config[k].keys()) + 1)]
            if type(v) != dict:
                is_valid = False
            elif sorted(config[k].keys()) != keys:
                is_valid = False
            else:
                for k1, v1 in v.items():
                    if type(k1) != int or type(v1) != int:
                        is_valid = False
        if k in ["learning_rate", "regularization"]:
            if type(v) != float:
                is_valid = False
        if k in ["prediction_confidence", "human_expertise"]:
            if type(v) != float or (v > 1 or v < 0):
                is_valid = False
        if k == "iterations":
            if type(v) != int:
                is_valid = False
        if k in ["seeded", "show_cost", "error_analysis", "init_random"]:
            if type(v) != bool:
                is_valid = False
        if k == "seed":
            if type(v) != int:
                is_valid = False
        if k == "activation":
            if type(v) != dict:
                is_valid = False
            elif not set(["relu", "sigmoid"]).issubset(v.values()):
                is_valid = False
        if not is_valid:
            logging.warning("Unexpected data type for the key {}".format(k))
            return False
    return True


def read_out_config(config_name):
    """ Wrapper for both YAML and JSON formats """
    extension = config_name.split(".")[-1]
    if extension == "json":
        try:
            conf = read_out_json_config(config_name)
            if is_valid_config(conf):
                return conf
            else:
                raise ValueError("Inappropriate config provided")
        except ValueError:
            raise ValueError("Inappropriate config provided")
    elif extension in ["yml", "yaml"]:
        try:
            conf = read_out_yaml_config(config_name)
            if is_valid_config(conf):
                return conf
            else:
                raise ValueError("Inappropriate config provided")
        except ValueError:
            raise ValueError("Inappropriate config provided")
    else:
        raise ValueError("Unknown config file extension")
