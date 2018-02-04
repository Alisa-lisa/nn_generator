from model_generator.simple_nn import SET_UP_KEYS
import json

def read_out_config(config_name):
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