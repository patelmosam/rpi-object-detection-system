import json

def get_config(config_file):
    with open(config_file) as cfg:
        data = json.load(cfg)

    return data