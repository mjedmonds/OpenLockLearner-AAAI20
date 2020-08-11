import os
import sys
import json

# project root dir, three directories up
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEBUGGING = True if "pydevd" in sys.modules else False


def load_json_config(path):
    with open(path) as json_data_file:
        config_data = json.load(json_data_file)
        return config_data
