"""
YAML/JSON config loader with dot-notation access.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict


class ConfigDict(dict):
    """Dict with dot-notation access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: str | Path) -> ConfigDict:
    """Load a YAML or JSON config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return ConfigDict(data)