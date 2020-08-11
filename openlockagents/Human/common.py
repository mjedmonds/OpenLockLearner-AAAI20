import json

import openlockagents.common.common as common


def load_human_config_json(path="openlockagents/Human/human_config.json"):
    return common.load_json_config(path)
