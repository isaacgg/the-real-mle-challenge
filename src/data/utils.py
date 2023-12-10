import json
import logging
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config: str) -> Dict:
    with open(config, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError:
            logger.error("Could not load config file", exc_info=True)


def save_json(data: Dict, file_path: str):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
